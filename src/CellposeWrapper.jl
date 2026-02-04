module CellposeWrapper

using PyCall
using Colors
using Plots

# --- GLOBALS ---
const cv2 = PyNULL()
const torch = PyNULL()
const py_model_loader = PyNULL()

# TRUE lazy flag
const _initialized = Ref(false)

# lock globale per rendere PyCall/Python thread-safe
const _py_lock = ReentrantLock()

# cache modelli (key stabile e tipizzata)
# key = (use_gpu::Bool, model_type::String, pretrained_model::String)
const _model_cache = Dict{Tuple{Bool,String,String},PyObject}()

function __init__()
    # Non inizializzare Python qui. Deve restare lazy.
    nothing
end

# --- helpers ---
_safe_array(x) = isa(x, PyObject) ? Array(x) : x
_is_py_none(x) = (x isa PyObject) && (py"lambda o: o is None"(x) == true)

# normalizza input per chiavi cache stabili
_norm_model_type(x) = (x === nothing || x == "") ? "cpsam" : String(x)
_norm_pretrained(x) = (x === nothing) ? "" : String(x)

# Converte un array HxWx3 UInt8/Float in Matrix{RGB{Float64}}
function _cv_rgb_to_rgbmat(img_rgb)
    a = _safe_array(img_rgb)  # tipicamente HxWx3 UInt8
    ndims(a) == 3 || error("Expected HxWxC array from cv2, got ndims=$(ndims(a)) size=$(size(a))")
    size(a, 3) == 3 || error("Expected 3 channels RGB, got C=$(size(a,3))")

    # a[i,j,1]=R a[i,j,2]=G a[i,j,3]=B (UInt8 0..255)
    if eltype(a) <: UInt8
        return [RGB{Float64}(a[i, j, 1] / 255, a[i, j, 2] / 255, a[i, j, 3] / 255) for i in 1:size(a, 1), j in 1:size(a, 2)]
    else
        # se è già float, assumiamo 0..1 oppure 0..255? qui assumo 0..1; se vedi immagini “scure”, cambia.
        return [RGB{Float64}(a[i, j, 1], a[i, j, 2], a[i, j, 3]) for i in 1:size(a, 1), j in 1:size(a, 2)]
    end
end

function _load_image_rgbmat_cv2(image_path::String)
    _init_py!()
    lock(_py_lock)
    try
        img_cv = cv2.imread(image_path)
        if _is_py_none(img_cv)
            error("Not Found or unreadable image: $image_path")
        end
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return _cv_rgb_to_rgbmat(img_rgb)
    finally
        unlock(_py_lock)
    end
end



"""
    clear_model_cache!()

Svuota la cache dei modelli. Utile se cambi spesso parametri o vuoi liberare memoria.
"""
function clear_model_cache!()
    lock(_py_lock)
    try
        empty!(_model_cache)
    finally
        unlock(_py_lock)
    end
    return nothing
end

function _init_py!()
    _initialized[] && return

    lock(_py_lock)
    try
        _initialized[] && return  # double-check

        @info "CellposeWrapper: Init Python (Lazy Mode)..."

        torch_temp = pyimport("torch")
        try
            if hasproperty(torch_temp.backends, "mps") && torch_temp.backends.mps.is_available()
                @info "CellposeWrapper: Mac Apple Silicon (MPS) active."
                torch_temp.set_default_dtype(torch_temp.float32)
            end
        catch
            # ignora, torch su alcune piattaforme non ha mps
        end

        py"""
        import os
        import cv2
        import torch
        from cellpose import models

        def _load_simple(gpu_on, model_type_str, custom_path):
            if model_type_str is None or model_type_str == "":
                model_type_str = "cpsam"

            if custom_path:
                weights_path = custom_path
            else:
                try:
                    weights_path = models.model_path(model_type_str)
                except Exception:
                    # fallback: assume sia un path o un identificatore valido
                    weights_path = model_type_str

            return models.CellposeModel(gpu=gpu_on, pretrained_model=weights_path)
        """

        copy!(cv2, py"cv2")
        copy!(torch, py"torch")
        copy!(py_model_loader, py"_load_simple")

        _initialized[] = true
        @info "CellposeWrapper: Python API configured."
    catch e
        _initialized[] = false
        @error "CellposeWrapper: CRITICAL INIT ERROR" exception = (e, catch_backtrace())
        rethrow()
    finally
        unlock(_py_lock)
    end
end


"""
    _get_model(use_gpu, model_type, pretrained_model; cache_models=true, max_cached_models=2)

- Normalizza già `model_type` e `pretrained_model` PRIMA di arrivare qui (quindi sono String stabili).
- Cache “general purpose”: evita reload continuo.
- max_cached_models limita la crescita della cache (svuota cache quando supera la soglia).
"""
function _get_model(
    use_gpu::Bool,
    model_type::String,
    pretrained_model::String;
    cache_models::Bool=true,
    max_cached_models::Int=2
)
    # no-cache: sempre un nuovo modello
    if !cache_models
        p_path = (pretrained_model == "") ? nothing : pretrained_model
        return py_model_loader(use_gpu, model_type, p_path)
    end

    key = (use_gpu, model_type, pretrained_model)
    m = get(_model_cache, key, nothing)

    if m === nothing
        # limite cache “generico”: evita crescita infinita
        if length(_model_cache) >= max_cached_models
            empty!(_model_cache)
        end

        p_path = (pretrained_model == "") ? nothing : pretrained_model
        m = py_model_loader(use_gpu, model_type, p_path)
        _model_cache[key] = m
    end

    return m
end

"""
    segment_image(image_path; kwargs...)

Ritorna:
- (masks=...,) se return_flows=false
- (masks=..., flows_rgb=..., cellprob=...) se return_flows=true

Opzioni migliorative:
- cache_models::Bool=true
- max_cached_models::Int=2
"""
function segment_image(image_path::String;
    diameter=nothing,
    model_type=nothing,
    pretrained_model=nothing,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    augment=false,
    invert=false,
    min_size=15,
    return_flows::Bool=false,
    cache_models::Bool=true,
    max_cached_models::Int=2
)
    _init_py!()

    # Tutto ciò che tocca Python va serializzato
    lock(_py_lock)
    try
        img_cv = cv2.imread(image_path)
        if _is_py_none(img_cv)
            error("Not Found or unreadable image: $image_path")
        end

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        use_gpu = false
        try
            use_gpu = torch.cuda.is_available()
        catch
        end
        try
            # mps può non esistere su non-mac
            use_gpu = use_gpu || (hasproperty(torch.backends, "mps") && torch.backends.mps.is_available())
        catch
        end

        # chiavi cache stabili
        mt = _norm_model_type(model_type)
        pm = _norm_pretrained(pretrained_model)

        model = _get_model(use_gpu, mt, pm; cache_models=cache_models, max_cached_models=max_cached_models)

        @info "CellposeWrapper: Analyzing..."

        params = Dict(
            :image_path => image_path,
            :diameter => diameter,
            :model_type => mt,
            :pretrained_model => pm,
            :flow_threshold => flow_threshold,
            :cellprob_threshold => cellprob_threshold,
            :augment => augment,
            :invert => invert,
            :min_size => min_size,
            :return_flows => return_flows,
            :cache_models => cache_models,
            :max_cached_models => max_cached_models,
            :use_gpu => use_gpu
        )
        @info "CellposeWrapper params" params = params

        results = model.eval(
            img_rgb,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            augment=augment,
            invert=invert,
            min_size=min_size
        )


        masks = _safe_array(results[1])

        if !return_flows
            est_diam = (length(results) >= 4) ? results[4] : 0.0
            if est_diam isa Real && est_diam > 0
                @info "CellposeWrapper: Done! Diameter: $(round(Float64(est_diam), digits=2)) px"
            else
                @info "CellposeWrapper: Done!"
            end
            return (masks=masks, params=params)
        end

        flows = results[2]
        flows_rgb = _safe_array(flows[1])
        cellprob = _safe_array(flows[3])

        est_diam = (length(results) >= 4) ? results[4] : 0.0
        if est_diam isa Real && est_diam > 0
            @info "CellposeWrapper: Done! Diameter: $(round(Float64(est_diam), digits=2)) px"
        else
            @info "CellposeWrapper: Done!"
        end

        return (masks=masks, flows_rgb=flows_rgb, cellprob=cellprob, params=params)
    finally
        unlock(_py_lock)
    end
end

"""
    show_results(results, image_path; view="masks")

view ∈ {"masks","flows","prob","image"}
"""
function show_results(results, image_path; view="masks")
    img = _load_image_rgbmat_cv2(image_path)
    n_cells = isempty(results.masks) ? 0 : maximum(results.masks)

    if view == "masks"
        masks = results.masks
        h, w = size(masks)
        @info "Visualization: $n_cells cells."

        overlay = fill(RGBA(0, 0, 0, 0), h, w)
        if n_cells > 0
            cols = distinguishable_colors(n_cells + 2, [RGB(0, 0, 0), RGB(1, 1, 1)])[3:end]
            @inbounds for i in 1:h, j in 1:w
                id = masks[i, j]
                if id > 0
                    c = cols[((id-1)%length(cols))+1]
                    overlay[i, j] = RGBA(c.r, c.g, c.b, 0.7)
                end
            end
        end

        p = plot(img, axis=false, title="Segmentation $n_cells cells")
        plot!(p, overlay, axis=false)
        display(p)

    elseif view == "prob"
        @info "Visualization Prob $n_cells cells"
        prob_map = results.cellprob
        display(heatmap(prob_map, axis=false, yflip=true, aspect_ratio=:equal,
            title="Probability $n_cells cells"))

    elseif view == "image"
        display(plot(img, axis=false))

    else
        @warn "Unknown view='$view'. Use 'masks', 'prob', or 'image'."
    end
end


export segment_image, show_results, clear_model_cache!
end # module
module CellposeWrapper

using PyCall
using Images
using Plots
using Colors
using FileIO

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

# normalizza input per chiavi cache stabili
_norm_model_type(x) = (x === nothing || x == "") ? "cyto" : String(x)
_norm_pretrained(x) = (x === nothing) ? "" : String(x)

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

        println("🔍 CellposeWrapper: Init Python (Lazy Mode)...")

        # torch import una sola volta
        torch_temp = pyimport("torch")
        if torch_temp.backends.mps.is_available()
            println("🚀 CellposeWrapper: Mac Apple Silicon (MPS) active.")
            torch_temp.set_default_dtype(torch_temp.float32)
        end

        py"""
        import os
        import cv2
        import torch
        from cellpose import models

        def _load_simple(gpu_on, model_type_str, custom_path):
            # robust default
            if model_type_str is None or model_type_str == "":
                model_type_str = "cyto"

            if custom_path:
                print(f"   --> Python: Using custom path: {custom_path}")
                weights_path = custom_path
            else:
                print(f"   --> Python: Requested '{model_type_str}'")
                try:
                    weights_path = models.model_path(model_type_str)
                except Exception as e:
                    print(f"   ⚠️ Warning path: {e}")
                    weights_path = model_type_str

            print(f"   --> Python: Loading from: {os.path.basename(str(weights_path))}")
            return models.CellposeModel(gpu=gpu_on, pretrained_model=weights_path)
        """

        copy!(cv2, py"cv2")
        copy!(torch, py"torch")
        copy!(py_model_loader, py"_load_simple")

        _initialized[] = true
        println("✅ Python API configured.")
    catch e
        println("❌ CRITICAL INIT ERROR")
        rethrow(e)
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
        if img_cv === nothing
            error("Not Found: $image_path")
        end
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        use_gpu = torch.cuda.is_available() || torch.backends.mps.is_available()

        # chiavi cache stabili
        mt = _norm_model_type(model_type)
        pm = _norm_pretrained(pretrained_model)

        model = _get_model(use_gpu, mt, pm; cache_models=cache_models, max_cached_models=max_cached_models)

        println("...Analyzing...")

        results = model.eval(
            img_rgb,
            diameter=diameter,
            channels=[0, 0],
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
                println("✅ Done! Diameter: $(round(Float64(est_diam), digits=2)) px")
            else
                println("✅ Done!")
            end
            return (masks=masks,)
        end

        flows = results[2]
        flows_rgb = _safe_array(flows[1])
        cellprob = _safe_array(flows[3])

        est_diam = (length(results) >= 4) ? results[4] : 0.0
        if est_diam isa Real && est_diam > 0
            println("✅ Done! Diameter: $(round(Float64(est_diam), digits=2)) px")
        else
            println("✅ Done!")
        end

        return (masks=masks, flows_rgb=flows_rgb, cellprob=cellprob)
    finally
        unlock(_py_lock)
    end
end

"""
    show_results(results, image_path; view="masks")

view ∈ {"masks","flows","prob","image"}
"""
function show_results(results, image_path; view="masks")
    # show_results è 100% Julia: NON serve lock Python.
    img = load(image_path)
    n_cells = maximum(results.masks)

    if view == "masks"
        masks = results.masks
        h, w = size(masks)
        println("Visualization: $n_cells cells.")

        overlay = fill(RGBA(0, 0, 0, 0), h, w)
        if n_cells > 0
            colors = distinguishable_colors(n_cells + 2, [RGB(0, 0, 0), RGB(1, 1, 1)])
            base_cols = colors[3:end]
            @inbounds for i in 1:h, j in 1:w
                id = masks[i, j]
                if id > 0
                    c_idx = ((id - 1) % length(base_cols)) + 1
                    c = base_cols[c_idx]
                    overlay[i, j] = RGBA(c.r, c.g, c.b, 0.4)
                end
            end
        end
        p = plot(img, axis=false, title="Segmentation $n_cells cells")
        plot!(p, overlay, axis=false)
        display(p)

    elseif view == "flows"
        println("Visualization Flows of $n_cells cells")
        flow_data = results.flows_rgb

        if ndims(flow_data) == 3
            if size(flow_data, 3) == 3
                if eltype(flow_data) <: UInt8
                    flow_img = colorview(RGB, permutedims(Float64.(flow_data) ./ 255.0, (3, 1, 2)))
                else
                    flow_img = colorview(RGB, permutedims(Float64.(flow_data), (3, 1, 2)))
                end
            elseif size(flow_data, 1) == 3
                if eltype(flow_data) <: UInt8
                    flow_img = colorview(RGB, Float64.(flow_data) ./ 255.0)
                else
                    flow_img = colorview(RGB, Float64.(flow_data))
                end
            else
                println("⚠️ Unrecognized flows format: $(size(flow_data))")
                return
            end
            display(plot(flow_img, axis=false, title="Flows $n_cells cells"))
        else
            println("⚠️ Flows have unexpected dims: $(ndims(flow_data))")
        end

    elseif view == "prob"
        println("Visualization Prob $n_cells cells")
        prob_map = results.cellprob
        display(heatmap(prob_map, axis=false, yflip=true, aspect_ratio=:equal, title="Probability $n_cells cells"))

    elseif view == "image"
        display(plot(img, axis=false))
    end
end

end # module
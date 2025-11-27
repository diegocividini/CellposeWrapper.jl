module CellposeWrapper

using PyCall
using Images
using Plots
using Colors

# --- GLOBALS ---
const cv2 = PyNULL()
const torch = PyNULL()
const py_model_loader = PyNULL()

function __init__()
    # 1. FIX MPS
    torch_temp = pyimport("torch")
    if torch_temp.backends.mps.is_available()
        println("🚀 CellposeWrapper: Mac Apple Silicon (MPS) attivo.")
        torch_temp.set_default_dtype(torch_temp.float32)
    end

    println("🔍 CellposeWrapper: Init Python (Simple Mode)...")

    try
        py"""
        import sys
        import os
        import cv2
        import torch
        from cellpose import models

        def _load_simple(gpu_on, model_type_str, custom_path):
            weights_path = None
            if custom_path:
                print(f"   --> Python: Uso custom path: {custom_path}")
                weights_path = custom_path
            else:
                print(f"   --> Python: Richiesto '{model_type_str}'")
                try:
                    weights_path = models.model_path(model_type_str)
                except Exception as e:
                    print(f"   ⚠️ Warning path: {e}")
                    weights_path = model_type_str

            print(f"   --> Python: Loading da: {os.path.basename(str(weights_path))}")
            return models.CellposeModel(gpu=gpu_on, pretrained_model=weights_path)
        """

        copy!(cv2, py"cv2")
        copy!(torch, py"torch")
        copy!(py_model_loader, py"_load_simple")

        println("✅ API Python configurata.")

    catch e
        println("❌ ERRORE CRITICO INIT")
        rethrow(e)
    end
end

# --- HELPER PER EVITARE ERRORI DI CONVERSIONE ---
# Converte in Array Julia solo se è ancora un PyObject.
# Se è già convertito, lo lascia stare.
function _safe_convert(obj)
    if isa(obj, PyObject)
        return PyArray(obj)
    else
        return obj
    end
end

"""
    segment_image(image_path; kwargs...)
"""
function segment_image(image_path::String;
    diameter=nothing,
    model_type="tissuenet",
    pretrained_model=nothing,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    augment=false,
    invert=false,
    min_size=15)

    img_cv = cv2.imread(image_path)
    if img_cv === nothing
        error("File non trovato: $image_path")
    end
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    use_gpu = torch.cuda.is_available() || torch.backends.mps.is_available()
    p_path = isnothing(pretrained_model) ? nothing : pretrained_model
    model = py_model_loader(use_gpu, model_type, p_path)

    println("...Analisi in corso...")

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

    masks = results[1]
    flows = results[2]

    # Estrazione dati grezzi
    flows_rgb_py = flows[1]
    cellprob_py = flows[3]

    est_diam = length(results) >= 4 ? results[4] : 0.0
    if est_diam > 0
        println("✅ Fatto! Diametro: $(round(est_diam, digits=2)) px")
    else
        println("✅ Fatto!")
    end

    # --- FIX DEFINITIVO ---
    # Usiamo la funzione sicura che controlla il tipo prima di convertire
    return (
        masks=_safe_convert(masks),
        flows_rgb=_safe_convert(flows_rgb_py),
        cellprob=_safe_convert(cellprob_py)
    )
end

"""
    show_results(results, image_path; view="masks")
"""
function show_results(results, image_path; view="masks")
    img = load(image_path)
    n_cells = maximum(results.masks)
    if view == "masks"
        masks = results.masks
        h, w = size(masks)
        println("Visualizzazione: $n_cells cellule.")

        overlay = fill(RGBA(0, 0, 0, 0), h, w)
        if n_cells > 0
            colors = distinguishable_colors(n_cells + 2, [RGB(0, 0, 0), RGB(1, 1, 1)])
            base_cols = colors[3:end]
            for i in 1:h, j in 1:w
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
        println("Visualizzazione Flows di $n_cells cellule")
        flow_data = results.flows_rgb
        # Python (H,W,C) -> Julia (C,W,H) di solito con PyCall standard
        # Se flow_data è già un array Julia, controlliamo le dimensioni

        # Logica adattiva per visualizzare l'immagine RGB
        # Se è un array 3D
        if ndims(flow_data) == 3
            # Se i canali sono alla fine (H,W,3), permutiamo
            if size(flow_data, 3) == 3
                if eltype(flow_data) <: UInt8
                    flow_img = colorview(RGB, permutedims(Float64.(flow_data) ./ 255.0, (3, 1, 2)))
                else
                    flow_img = colorview(RGB, permutedims(Float64.(flow_data), (3, 1, 2)))
                end
                # Se i canali sono all'inizio (3,W,H) - PyCall a volte fa questo
            elseif size(flow_data, 1) == 3
                if eltype(flow_data) <: UInt8
                    flow_img = colorview(RGB, Float64.(flow_data) ./ 255.0)
                else
                    flow_img = colorview(RGB, Float64.(flow_data))
                end
            else
                println("⚠️ Formato flows non riconosciuto: $(size(flow_data))")
                return
            end

            display(plot(flow_img, axis=false, title="Flows $n_cells cells"))
        end

    elseif view == "prob"
        println("Visualizzazione Prob $n_cells cellule")
        prob_map = results.cellprob
        display(heatmap(prob_map, c=:inferno, axis=false, yflip=true, aspect_ratio=:equal, title="Probability $n_cells cells"))

    elseif view == "image"
        display(plot(img, axis=false))
    end
end

end
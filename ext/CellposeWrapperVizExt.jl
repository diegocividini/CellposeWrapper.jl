module CellposeWrapperVizExt

# Extension for visualization utilities (masks / flows / prob / image)

using CellposeWrapper
using Plots
using Colors
using FileIO
using Images
using PyCall

# -------------------------
# helpers (local, robust)
# -------------------------

_safe_array(x) = isa(x, PyObject) ? Array(x) : x

# Prefer wrapper's helpers if they exist, else fall back.
const _has_wrapper_safe_array = isdefined(CellposeWrapper, :_safe_array)
const _has_wrapper_init_py = isdefined(CellposeWrapper, :_init_py!)
const _has_wrapper_lock = isdefined(CellposeWrapper, :_py_lock)
const _has_wrapper_cv2 = isdefined(CellposeWrapper, :cv2)

# Small helper to detect Python None robustly.
# If your core has a faster/cleaner function, you can swap it in.
function _is_py_none(x)
  # Use a tiny python identity check without re-defining it.
  # (This only runs when you call show_results.)
  return (x isa PyObject) && (py"lambda o: o is None"(x) == true)
end

# Convert cv2 RGB image (HxWx3 UInt8/Float) into Matrix{RGB{Float64}}
function _cv_rgb_to_rgbmat(img_rgb)
  a = _has_wrapper_safe_array ? CellposeWrapper._safe_array(img_rgb) : _safe_array(img_rgb)

  ndims(a) == 3 || error("Expected HxWxC array from cv2, got ndims=$(ndims(a)) size=$(size(a))")
  size(a, 3) == 3 || error("Expected 3 channels RGB, got C=$(size(a,3))")

  h, w = size(a, 1), size(a, 2)
  out = Matrix{RGB{Float64}}(undef, h, w)

  if eltype(a) <: UInt8
    @inbounds for i in 1:h, j in 1:w
      out[i, j] = RGB{Float64}(a[i, j, 1] / 255.0, a[i, j, 2] / 255.0, a[i, j, 3] / 255.0)
    end
  else
    # assume float in 0..1 (if it's 0..255 you'll see a blown-out image)
    @inbounds for i in 1:h, j in 1:w
      out[i, j] = RGB{Float64}(Float64(a[i, j, 1]), Float64(a[i, j, 2]), Float64(a[i, j, 3]))
    end
  end
  return out
end

# Load image for visualization.
# Preferred: FileIO.load (pure Julia).
# Fallback: cv2 (if FileIO cannot load).
function _load_image_rgbmat(image_path::AbstractString)
  # Try Julia loader first (fast, no Python needed)
  try
    img = load(String(image_path))
    return img
  catch
    # fallback to cv2 if available
  end

  _has_wrapper_init_py || error("CellposeWrapper._init_py! not available; cannot init python fallback loader.")
  _has_wrapper_lock || error("CellposeWrapper._py_lock not available; cannot lock python fallback loader.")
  _has_wrapper_cv2 || error("CellposeWrapper.cv2 not available; cannot read via OpenCV.")

  CellposeWrapper._init_py!()
  lock(CellposeWrapper._py_lock)
  try
    img_cv = CellposeWrapper.cv2.imread(String(image_path))
    if _is_py_none(img_cv)
      error("Not Found or unreadable image: $image_path")
    end
    img_rgb = CellposeWrapper.cv2.cvtColor(img_cv, CellposeWrapper.cv2.COLOR_BGR2RGB)
    return _cv_rgb_to_rgbmat(img_rgb)
  finally
    unlock(CellposeWrapper._py_lock)
  end
end

# Convert flows array to something plottable (RGB image)
function _flows_to_rgb(flow_data)
  a = _has_wrapper_safe_array ? CellposeWrapper._safe_array(flow_data) : _safe_array(flow_data)

  if ndims(a) != 3
    error("Flows have unexpected dims: ndims=$(ndims(a)) size=$(size(a))")
  end

  # Common formats:
  # - HxWx3 (cv2-style) or HxWxC
  # - 3xHxW (channel-first)
  if size(a, 3) == 3
    # HxWx3
    if eltype(a) <: UInt8
      return colorview(RGB, permutedims(Float64.(a) ./ 255.0, (3, 1, 2)))
    else
      return colorview(RGB, permutedims(Float64.(a), (3, 1, 2)))
    end
  elseif size(a, 1) == 3
    # 3xHxW
    if eltype(a) <: UInt8
      return colorview(RGB, Float64.(a) ./ 255.0)
    else
      return colorview(RGB, Float64.(a))
    end
  else
    error("Unrecognized flows format: size=$(size(a)) (expected HxWx3 or 3xHxW)")
  end
end

# -------------------------
# Public extension method
# -------------------------

"""
    CellposeWrapper.show_results(results, image_path; view="masks")

`view ∈ {"masks","flows","prob","image"}`

- "masks": overlay colored instance masks on the input image
- "flows": show Cellpose flow visualization (requires `return_flows=true`)
- "prob": show Cellpose cell probability map (requires `return_flows=true`)
- "image": show only the input image
"""
function CellposeWrapper.show_results(results, image_path::AbstractString; view::AbstractString="masks")
  img = _load_image_rgbmat(image_path)

  masks = getproperty(results, :masks)
  n_cells = isempty(masks) ? 0 : maximum(masks)

  if view == "masks"
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
    return nothing

  elseif view == "flows"
    @info "Visualization Flows of $n_cells cells"
    if !hasproperty(results, :flows_rgb)
      @warn "results.flows_rgb not found. Call segment_image(...; return_flows=true)."
      return nothing
    end
    flow_rgb = _flows_to_rgb(getproperty(results, :flows_rgb))
    display(plot(flow_rgb, axis=false, title="Flows $n_cells cells"))
    return nothing

  elseif view == "prob"
    @info "Visualization Prob $n_cells cells"
    if !hasproperty(results, :cellprob)
      @warn "results.cellprob not found. Call segment_image(...; return_flows=true)."
      return nothing
    end
    prob_map = getproperty(results, :cellprob)
    display(heatmap(prob_map, axis=false, yflip=true, aspect_ratio=:equal, title="Probability $n_cells cells"))
    return nothing

  elseif view == "image"
    display(plot(img, axis=false))
    return nothing

  else
    @warn "Unknown view='$view'. Use 'masks', 'flows', 'prob', or 'image'."
    return nothing
  end
end

end # module

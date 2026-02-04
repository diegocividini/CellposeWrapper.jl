module CellposeWrapperVizExt

using CellposeWrapper
using Images
using Plots
using Colors
using FileIO

function CellposeWrapper.show_results(results, image_path; view="masks")
  img = load(image_path)
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

  elseif view == "flows"
    @info "Visualization Flows of $n_cells cells"
    flow_data = results.flows_rgb
    # (tua logica invariata)
    # ...
  elseif view == "prob"
    @info "Visualization Prob $n_cells cells"
    prob_map = results.cellprob
    display(heatmap(prob_map, axis=false, yflip=true, aspect_ratio=:equal, title="Probability $n_cells cells"))
  elseif view == "image"
    display(plot(img, axis=false))
  end
end

end

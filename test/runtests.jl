using Test
using CellposeWrapper

const _HAVE_PYCALL = let ok = true
  try
    @eval using PyCall
  catch
    ok = false
  end
  ok
end
@testset "CellposeWrapper - basic package load" begin
  @test isdefined(CellposeWrapper, :segment_image)
  @test isdefined(CellposeWrapper, :clear_model_cache!)
  @test isdefined(CellposeWrapper, :init!)
  @test isdefined(CellposeWrapper, :show_results)

  @test isa(CellposeWrapper.show_results, Function)

  # Extension should not be loaded by default
  @test Base.get_extension(CellposeWrapper, :CellposeWrapperVizExt) === nothing

  @test CellposeWrapper.clear_model_cache!() === nothing
end

@testset "CellposeWrapper - optional viz extension (if deps installed)" begin
  have_plots = Base.find_package("Plots") !== nothing
  have_colors = Base.find_package("Colors") !== nothing
  have_fileio = Base.find_package("FileIO") !== nothing
  have_images = Base.find_package("Images") !== nothing

  if have_plots && have_colors && have_fileio && have_images
    @eval using Plots
    @eval using Colors
    @eval using FileIO
    @eval using Images

    ext = Base.get_extension(CellposeWrapper, :CellposeWrapperVizExt)
    @test ext !== nothing
    @test length(methods(CellposeWrapper.show_results)) > 0
  else
    @test true
  end
end

@testset "CellposeWrapper - optional Cellpose runtime smoke test" begin
  if !_HAVE_PYCALL
    @test true
    return
  end

  function py_has(mod::AbstractString)
    try
      PyCall.pyimport(mod)
      return true
    catch
      return false
    end
  end

  has_cv2 = py_has("cv2")
  has_torch = py_has("torch")
  has_cellpose = py_has("cellpose")

  if !(has_cv2 && has_torch && has_cellpose)
    @test true
    return
  end

  @test begin
    try
      CellposeWrapper.init!()
      true
    catch err
      @info "CellposeWrapper.init! failed although python deps seem present" err
      false
    end
  end

  tmp = mktempdir()
  imgpath = joinpath(tmp, "tiny.png")

  PyCall.py"""
import numpy as np, cv2
img = np.zeros((128,128,3), dtype=np.uint8)
cv2.circle(img, (64,64), 18, (255,255,255), -1)
cv2.imwrite($imgpath, img)
"""

  res = CellposeWrapper.segment_image(imgpath; return_flows=false)
  @test hasproperty(res, :masks)
  @test size(res.masks, 1) > 0 && size(res.masks, 2) > 0
end


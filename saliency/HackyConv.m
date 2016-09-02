classdef HackyConv < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
    scale = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = vl_nnconv(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias
         params{2} = [] ; 
         derParams{2} = [];
         derIntermediate = derOutputs{1};
      else
         % First the bias correction
         theeye = reshape(ones(size(params{1}, 4), 1, 'single'), ...
                     [1, 1, 1, size(params{1}, 4)]);
         if(isa(params{1}, 'gpuArray'))
           theeye = gpuArray(theeye);
         end
         derIntermediate = vl_nnconv(derOutputs{1},...
           theeye, -params{2}, 'pad', 0, 'stride', 1, obj.opts{:});
         derParams{2} = [];
      end

      % Next the actual deconvolution
      [derInputs{1}, derParams{1}, ~] = vl_nnconv(...
        inputs{1}, params{1}, [], derIntermediate, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;

      % scale correction
      derInputs{1} = derInputs{1} / obj.scale;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') * sc ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = Conv(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end

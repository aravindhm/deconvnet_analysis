classdef HackyReLU < dagnn.ElementWise
  properties
    leak = 0
    opts = {}
    hackType = 'none'
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnrelu(inputs{1}, [], ...
                             'leak', obj.leak, obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnrelu(inputs{1}, derOutputs{1}, ...
                               'leak', obj.leak, ...
                               obj.opts{:}) ;
      derParams = {} ;
      switch lower(obj.hackType)
        case 'us'
          derInputs{1} = max(derInputs{1}, 0) ;
        case 'deconvnet'
          derInputs{1} = max(derOutputs{1}, 0) ;
        case 'none'
        otherwise
          assert(false) ;
      end
    end

    function obj = HackyReLU(varargin)
      obj.load(varargin) ;
    end
  end
end

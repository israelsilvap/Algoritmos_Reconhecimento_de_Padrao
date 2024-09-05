classdef TreeNode
    properties
        Data
        Left
        Right
    end
    
    methods
        function obj = TreeNode(data)
            if nargin > 0
                obj.Data = data;
            end
            obj.Left = [];
            obj.Right = [];
        end
    end
end
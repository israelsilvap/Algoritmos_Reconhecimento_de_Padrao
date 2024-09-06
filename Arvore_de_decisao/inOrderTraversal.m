function inOrderTraversal(node)
    if ~isempty(node)
        inOrderTraversal(node.Left);
        disp(node.Data);
        inOrderTraversal(node.Right);
    end
end
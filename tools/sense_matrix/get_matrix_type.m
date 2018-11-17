function mat_type = get_matrix_type( mtx )
  %get_matrix_type Retruns a string representing the type of a matrix mtrx
  %
  %  mtx can be a SensingMatrix object a CS_EncParams object, or an struct having
  %     used to define a matrix, or a cell array or struct array with fields 
  %     'type' and possibly 'args'
  
  sub_mtx = [];
  
  if iscell(mtx)
    mat_type = cell(size(mtx));
    for k=1:numel(mtx)
      mat_type{k} = get_matrix_type(mtx{k});
    end
  elseif isa(mtx, 'SensingMatrix')
    mat_type = regexprep(class(mtx),'^SensingMatrix', '');
    if isa(mat_type, 'SensingMatrixComposed')
      sub_mtx = cell(2,numel(mtx.mtrx));
      for k=1:numel(mtx.mtrx)
        sub_mtx{1,k} = get_matrix_type(mtx.mtrx{k});
      end
    end
  elseif isa(mtx, 'CSEncParams')
    mat_type = regexprep(mtx.msrmnt_mtrx.type,'^SensingMatrix', '');
    if isfield(mtx.msrmnt_mtrx, 'args') && isfield(mtx.msrmnt_mtrx.args, 'mtrcs')
      sub_mtx = cell(2,numel(msrmnt_mtrx.args.mtrcs));
      for k=1:numel(sub_mtx)
        sub_mtx{1,k} = get_matrix_type(msrmnt_mtrx.args.mtrcs{k});
      end
    end
  elseif numel(mtx) > 1
    mat_type = cell(size(mtx));
    for k=1:numel(mtx)
      mat_type{k} = get_matrix_type(mtx(k));
    end
  else
    if isfield(mtx, 'msrmnt_mtrx')
      mtx = mtx.msrmnt_mtrx;
    end
    mat_type = regexprep(mtx.type,'^SensingMatrix', '');
    if isfield(mtx, 'args') && isfield(mtx.args, 'mtrcs')
      sub_mtx = cell(2,numel(mtx.args.mtrcs));
      for k=1:numel(mtx.args.mtrcs)
        sub_mtx{1,k} = get_matrix_type(mtx.args.mtrcs{k});
      end
    end
  end
  if ~isempty(sub_mtx)
    sub_mtx(2,:)={','};
    sub_mtx{2,end} = '';
    mat_type = [mat_type, '[', horzcat(sub_mtx{:}), ']' ];
  end
end


function mat_args = get_matrix_args( mtx )
  %get_matrix_args Retruns a string representing a atring which represent the 
  %args of a matrix mtrx
  %
  %  mtx can be a CS_EncParams object, or an struct having
  %     used to define a matrix, or a cell array or struct array with fields 
  %     'type' and possibly 'args'
  
  sub_mtx = [];
  
  if iscell(mtx)
    mat_args = cell(size(mtx));
    for k=1:numel(mtx)
      mat_args{k} = get_matrix_args(mtx{k});
    end
  elseif isa(mtx, 'CSEncParams')
    mat_args = get_matrix_args(mtx.msrmnt_mtrx);
  elseif numel(mtx) > 1
    mat_args = cell(size(mtx));
    for k=1:numel(mtx)
      mat_args{k} = get_matrix_args(mtx(k));
    end
  else
    if isfield(mtx, 'msrmnt_mtrx')
      mtx = mtx.msrmnt_mtrx;
    end
    if ~isfield(mtx,'args') || isempty(mtx.args)
      mat_args = '{}';
    else
      if isfield(mtx.args, 'mtrcs')
        mtrcs = mtx.args.mtrcs;
        mtx.args = rmfield(mtx.args, 'mtrcs');
        sub_mtx = cell(2,numel(mtrcs));
        for k=1:numel(mtrcs)
          sub_mtx{1,k} = get_matrix_args(mtrcs{k});
        end
      end
      mat_args = mat2json(mtx.args, '', true);
      mat_args = regexprep(mat_args, '["]','');
    end
    if ~isempty(sub_mtx)
      sub_mtx(2,:)={','};
      sub_mtx{2,end} = '';
      mat_args = [mat_args, '[', horzcat(sub_mtx{:}), ']' ];
    end
  end
end



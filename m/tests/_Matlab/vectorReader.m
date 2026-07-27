function vector = vectorReader(filename)

fid = fopen(filename, 'r');
fscanf(fid, '%s', 1);
type = fscanf(fid, '%s', 1);
rowDim  = fscanf(fid, '%i', 1);

data = textscan(fid,'%f','Whitespace',' \b\t\n(,)');
fclose(fid);

switch type
  case 'AR'
    vector=data{1};
  case 'AC'
    vector = data{1}(1:2:end) + 1i*data{1}(2:2:end);
  case 'AI'
    vector=data{1};
  otherwise
    error('not yet implemented. Wrong array type in reader');
end

if(length(vector)~=rowDim)
  error('vector length does not match rowDim');
end

% ATTENTION: sparse vectors causes Matlab to crash during ROM generation!!!
% if(nnz(vector)<rowDim/3)
%   vector=sparse(vector);
% end

function writeMatFull(Mat, filename)

fid = fopen( filename, 'wt' );
if fid == -1
  error(strcat('Could not open file: ', filename));
end

typeString = 'M';
% determine type of the matrix
if isreal(Mat)  % matrix is real
  typeString = strcat(typeString,'R');
else
  typeString = strcat(typeString,'C');
end

typeString = strcat(typeString,'F');  % matrix is full

[r c] = size(Mat);
if r == c % matrix is quadratic
  if nnz(Mat-Mat.') == 0  % matrix is symmetric
    typeString = strcat(typeString,'S');
  else
    typeString = strcat(typeString,'Q');
  end
else
  typeString = strcat(typeString,'R');
end

fprintf(fid, '{ '); % write '{' in file
fprintf(fid, '%s ', typeString);
if typeString(4) == 'R'
  fprintf(fid, '%i %i \n', r, c);
else
  fprintf(fid, '%i \n', r);
end

cnt = 0;
if typeString(4) == 'S'
  if typeString(2) == 'C'
    for col = 1:r
      for row = 1:col
        fprintf(fid, '(%18.17e, %18.17e) ', real(Mat(row,col)), imag(Mat(row,col)));
        cnt = cnt+1;
        if mod(cnt,2) == 0
          fprintf(fid, '\n');
        end
      end
    end
  elseif typeString(2) == 'R'
    for col = 1:r
      for row = 1:col
        fprintf(fid, '%18.17e ', Mat(row,col));
        cnt = cnt+1;
        if mod(cnt,2) == 0
          fprintf(fid, '\n');
        end
      end
    end
  else
    error('Something is wrong with the matrix type');
  end
elseif typeString(4) == 'R' | typeString(4) == 'Q'
  if typeString(2) == 'C'
    for row = 1:r
      for col = 1:c
        fprintf(fid, '(%18.17e, %18.17e) ', real(Mat(row,col)), imag(Mat(row,col)));
        cnt = cnt+1;
        if mod(cnt,2) == 0
          fprintf(fid, '\n');
        end
      end
    end
  elseif typeString(2) == 'R'
    for row = 1:r
      for col = 1:c
        fprintf(fid, '%18.17e ', Mat(row,col));
        cnt = cnt+1;
        if mod(cnt,2) == 0
          fprintf(fid, '\n');
        end
      end
    end
  else
    error('Something is wrong with the matrix type');
  end
else
  error('Not yet implemented');
end
fprintf(fid, '}');

fclose(fid);

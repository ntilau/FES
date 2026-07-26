function writeVector(vec, filename)

fid = fopen( filename, 'w' );

[r c] = size(vec);

if r == 1   % row vector
  vec = vec.';
end

if isreal(vec)  % vector is real
  fprintf(fid, '{ AR %i \n', length(vec));
  fprintf(fid, ' %18.16e\n', vec);
else
  fprintf(fid, '{ AC %i \n', length(vec));
  fprintf(fid, '(%18.16e, %18.16e)\n', [real(vec)'; imag(vec)']);
end
fprintf( fid, ' }' );

fclose( fid );

function comsol = comsolReader(filename)
% Read a Comsol mph mesh file (.mphtxt format).
fid = fopen(filename, 'r');
tline = fgetl(fid);
iType = -1;
while ischar(tline)
    if strfind(tline, '# number of mesh points')
        pos = strfind(tline, '# number of mesh points');
        nnodes = str2double(tline(1:pos-1));
    end
    if strfind(tline, '# Mesh point coordinates')
        for i = 1:nnodes
            tline = fgetl(fid);
            comsol.nodes.mat(:, i) = sscanf(tline, '%g');
        end
    end

    if strfind(tline, '# number of element types')
        pos = strfind(tline, '# number of element types');
        comsol.eletypes = str2double(tline(1:pos-1));
    end

    if strfind(tline, '# Type')
        iType = iType + 1;
    end

    if strfind(tline, '# number of nodes per element')
        pos = strfind(tline, '# number of nodes per element');
        nnodes = str2double(tline(1:pos-1));
        comsol.elenodes(iType) = nnodes;
    end
    if strfind(tline, '# number of elements')
        pos = strfind(tline, '# number of elements');
        neles = str2double(tline(1:pos-1));
    end
    if strfind(tline, '# Elements')
        for i = 1:neles
            tline = fgetl(fid);
            comsol.ele{iType}.mat(:, i) = sscanf(tline, '%g');
        end
    end

    if strfind(tline, '# number of parameter values per element')
        pos = strfind(tline, '# number of parameter values per element');
        nparamval = str2double(tline(1:pos-1));
        comsol.paramvalues(iType) = nparamval;
    end
    if strfind(tline, '# number of parameters')
        pos = strfind(tline, '# number of parameters');
        nparams = str2double(tline(1:pos-1));
    end
    if strfind(tline, '# Parameters')
        for i = 1:nparams
            tline = fgetl(fid);
            comsol.par{iType}.mat(:, i) = sscanf(tline, '%g');
        end
    end

    if strfind(tline, '# number of geometric entity indices')
        pos = strfind(tline, '# number of geometric entity indices');
        ngeom = str2double(tline(1:pos-1));
    end
    if strfind(tline, '# Geometric entity indices')
        for i = 1:ngeom
            tline = fgetl(fid);
            comsol.geom{iType}.mat(:, i) = sscanf(tline, '%g');
        end
    end

    if strfind(tline, '# number of up/down pairs')
        pos = strfind(tline, '# number of up/down pairs');
        nudpairs = str2double(tline(1:pos-1));
    end
    if strfind(tline, '# Up/down')
        for i = 1:nudpairs
            tline = fgetl(fid);
            comsol.udpairs{iType}.mat(:, i) = sscanf(tline, '%g');
        end
    end

    tline = fgetl(fid);
end

fclose(fid);

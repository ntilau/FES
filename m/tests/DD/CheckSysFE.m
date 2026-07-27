format longe
db = [-8.7486e-011 -1.0696e+002];
db2 = [-3.0940e-006 -1.0759e+002];
abs = 10.^(db/20);
abs2 = 10.^(db2/20);
norm(abs)
norm(abs2)

# geometry_clustering
A Machine Learning Project: Based on geometry similarity (secondary or supersecondary structures of biomolecules) to divide structures into clusters to eleminate "duplicate" structures and also sort by Hydrogen-Bonding Pattern and total electronic energy at current computation level

Usage `$python cluster_kmean_hb.py energyFile keyAtomFile maxK`

## input file types
`energyFile` : tab or white space deliminated two column text(without header)-- column1: `.com` filenames, column2: current energy
  This file was used mainly to get a list of file of interest, the energy are at the current level of computation.
  
`keyAtomFile` : a chemistry concept, key atoms that define the 'back-bone' and chemical conformation of a biomolecule, such as $\alpha$-C and amide Nitrogens. Two key atoms defined a linear structure, and every set of three not-colinear atoms of a molecule defines a plane.
  Only limited number of atoms are "informative" in determining the structures so only those are used as key atoms
  format : `,` eliminated atom numbers, can be one line or multiple line
  Could be generated by pasting from `gaussview atom tools` > `atom selection` result and unpact some '-' into ','

`maxK` : max cluster number considered for all the geometries, depends on use cases
  For initial sorting(clustering) from Molecular Dynamics trajectorys, max K was ususally assigned as 1/4 to 1/3 of the geometry sample size if generated from one initial structure. Given the nature of our prior data cleaning procedure:(1) 1/100 snapshots shrinking initially (get one snapshot per 100 fs) from trajectory and (2) rough geometry optimization(valence shell electron method PM6-D3H4), structures are brought to their local minimum, sample size are shrinked but duplicates are unavoidable. Sufficient max K is required to observe an "elbow" in plotting the `RMSD` vs. `number-of-cluster` graph. 
  For secondary and up sorting(clustering) from next level of computation based on shrinked sample size generated from first sorting, `maxK` usually was used as the number of sample size, an "elbow" might not be observed with these levels.
  
`geometry files` : geometry files are generated from previous data cleaning steps that are `.com` gaussview readable files and have certain unstructured headlines and structured part `atomtype` `x` `y` `z` organized by atom number order and usually ome empty tail lines

## how it works
### Hydrogen Bonding
Automated generate qualified H-bonding (X-H_Y) list of each geometry without having to consider atom orders, where X, Y are N, O or F atom types and indicated X-H is covalent linked and H-bonded to Y(basic acceptor). In the ouput and intermediate files, each file(geometry) has a list of H-bond denoted as Xi-Hj_Yk. Note Xi-Hj are unique in each geometry (biomolecules) but Yk can have more than one. 
Hydrogen Bonding pattern was then 'ID'ed into a factor to allow being used for further sorting and grouping but was not used as input for K-mean clustering
#### Criterian of H-Bonding
X,Y has to be from N, O, F (very basic Carbon required a lot more info therefore was excluded in this script)
bond length X-H has to be qualified for covalent bond
bond length H-Y has to be qualified for H-bond length
bond angle X-H-Y has to be equal to or bigger than 120&deg;

### Unsupervised learning
We don't have a training set to know how many clusters are present in the sample set

### Use Distance Instead of Coordinates
This "alignment free" clustering script used `keyAtomFile` generated combination of every two key atom and get corresponding distances as nd array of "internal distance pairs" to represent each geometry, without having to align each geometry.
This approach would eliminate the effect of rotation and translation of the molecule (will be considered as same structure) since their relative (internal) atom positions remains unchanged.

### K mean clustering
#### input
use nd array of "internal distance pairs" as input for k mean clustering 
#### which k to use
try k from 1 to `maxK` and record the `RMSD to stablized centroid` and `score` generate by `sklean`
plot an `elbow plot` of RMSD and score vs. k value and find the corresponding k of the data(k, rmsd) represented the "elbow point"
"elbow point" was determined by finding the furthest point of the curve to the line(x0, xn)
Then use this `best K` for a final kmean clustering and record the cluster details.


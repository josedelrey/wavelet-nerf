mkdir -p datasets
cd datasets
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip nerf_example_data.zip
rm nerf_example_data.zip
rm -rf nerf_llff_data
mv nerf_synthetic/lego .
rm -rf nerf_synthetic
cd ..

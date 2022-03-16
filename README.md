Wed 16 Mar 2022 12:29:44 PM EDT


## Install

python3 and packages required can be installed system wide or a conda env

I suggest using conda:  
Install miniconda or anaconda normally if not already installed  

```
conda create -n outbreak-reporter
conda activate outbreak-reporter
conda install pandas numpy matplotlib openpyxl scipy pytables
conda install -c conda-forge openjdk parallel
```

Conda package for texlive is currently not working quite right, so need to install systemwide  
`sudo apt install texlive-full`  
Other latex installtions will probably work fine if they provide pdflatex




## Current workflow:
`conda activate outbreak-reporter`
`cp -r skel <short_name>`  
`cd <short_name>`  
`mkdir official`  
copy any SITSTAT or lifecyle docs to official directory  
edit main.cfg  
`../bin/fetch_and_process_temperatures.py`  
`../bin/setup_medfoes.py`  
`cd medfoes`  
`./run_mfp_local.sh`  
wait for medfoes runs to complete  
`cd ..`  
`../bin/make_report.py`  
report.pdf is the final output  


To update...
start from:
`../bin/fetch_and_process_temperatures.py`  
if only current year needs update, can run
`./run_mfp_current.sh` instead of `./run_mfp_local.sh` from the `medfoes` directory

To run on a cluster using SGE, 
run `../../bin/do_mfp_runs_SGE.sh` instead of `./run_mfp_local.sh` from the `medfoes` directory



#### Tested conda environment
```
# packages in environment at <snip>/miniconda3/envs/outbreak-reporter:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       1_gnu    conda-forge
alsa-lib                  1.2.3                h516909a_0    conda-forge
blosc                     1.21.0               h9c3ff4c_0    conda-forge
brotli                    1.0.9                h7f98852_6    conda-forge
brotli-bin                1.0.9                h7f98852_6    conda-forge
bzip2                     1.0.8                h7f98852_4    conda-forge
c-ares                    1.18.1               h7f98852_0    conda-forge
ca-certificates           2021.10.8            ha878542_0    conda-forge
certifi                   2021.10.8       py310hff52083_1    conda-forge
cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
dbus                      1.13.6               h5008d03_3    conda-forge
et_xmlfile                1.0.1                   py_1001    conda-forge
expat                     2.4.7                h27087fc_0    conda-forge
font-ttf-dejavu-sans-mono 2.37                 hab24e00_0    conda-forge
font-ttf-inconsolata      3.000                h77eed37_0    conda-forge
font-ttf-source-code-pro  2.038                h77eed37_0    conda-forge
font-ttf-ubuntu           0.83                 hab24e00_0    conda-forge
fontconfig                2.13.96              h8e229c2_2    conda-forge
fonts-conda-ecosystem     1                             0    conda-forge
fonts-conda-forge         1                             0    conda-forge
fonttools                 4.30.0          py310h5764c6d_0    conda-forge
freetype                  2.10.4               h0708190_1    conda-forge
gettext                   0.19.8.1          h73d1719_1008    conda-forge
giflib                    5.2.1                h36c2ea0_2    conda-forge
gst-plugins-base          1.18.5               hf529b03_3    conda-forge
gstreamer                 1.18.5               h9f60fe5_3    conda-forge
hdf5                      1.12.1          nompi_h2386368_104    conda-forge
icu                       69.1                 h9c3ff4c_0    conda-forge
jbig                      2.1               h7f98852_2003    conda-forge
jpeg                      9e                   h7f98852_0    conda-forge
keyutils                  1.6.1                h166bdaf_0    conda-forge
kiwisolver                1.4.0           py310hbf28c38_0    conda-forge
krb5                      1.19.3               h3790be6_0    conda-forge
lcms2                     2.12                 hddcbb42_0    conda-forge
ld_impl_linux-64          2.36.1               hea4e1c9_2    conda-forge
lerc                      3.0                  h9c3ff4c_0    conda-forge
libblas                   3.9.0           13_linux64_openblas    conda-forge
libbrotlicommon           1.0.9                h7f98852_6    conda-forge
libbrotlidec              1.0.9                h7f98852_6    conda-forge
libbrotlienc              1.0.9                h7f98852_6    conda-forge
libcblas                  3.9.0           13_linux64_openblas    conda-forge
libclang                  13.0.1          default_hc23dcda_0    conda-forge
libcurl                   7.82.0               h7bff187_0    conda-forge
libdeflate                1.10                 h7f98852_0    conda-forge
libedit                   3.1.20191231         he28a2e2_2    conda-forge
libev                     4.33                 h516909a_1    conda-forge
libevent                  2.1.10               h9b69904_4    conda-forge
libffi                    3.4.2                h7f98852_5    conda-forge
libgcc-ng                 11.2.0              h1d223b6_14    conda-forge
libgfortran-ng            11.2.0              h69a702a_14    conda-forge
libgfortran5              11.2.0              h5c6108e_14    conda-forge
libglib                   2.70.2               h174f98d_4    conda-forge
libgomp                   11.2.0              h1d223b6_14    conda-forge
libiconv                  1.16                 h516909a_0    conda-forge
liblapack                 3.9.0           13_linux64_openblas    conda-forge
libllvm13                 13.0.1               hf817b99_2    conda-forge
libnghttp2                1.47.0               h727a467_0    conda-forge
libnsl                    2.0.0                h7f98852_0    conda-forge
libogg                    1.3.4                h7f98852_1    conda-forge
libopenblas               0.3.18          pthreads_h8fe5266_0    conda-forge
libopus                   1.3.1                h7f98852_1    conda-forge
libpng                    1.6.37               h21135ba_2    conda-forge
libpq                     14.2                 hd57d9b9_0    conda-forge
libssh2                   1.10.0               ha56f1ee_2    conda-forge
libstdcxx-ng              11.2.0              he4da1e4_14    conda-forge
libtiff                   4.3.0                h542a066_3    conda-forge
libuuid                   2.32.1            h7f98852_1000    conda-forge
libvorbis                 1.3.7                h9c3ff4c_0    conda-forge
libwebp                   1.2.2                h3452ae3_0    conda-forge
libwebp-base              1.2.2                h7f98852_1    conda-forge
libxcb                    1.13              h7f98852_1004    conda-forge
libxkbcommon              1.0.3                he3ba5ed_0    conda-forge
libxml2                   2.9.12               h885dcf4_1    conda-forge
libzlib                   1.2.11            h36c2ea0_1013    conda-forge
lz4-c                     1.9.3                h9c3ff4c_1    conda-forge
lzo                       2.10              h516909a_1000    conda-forge
matplotlib                3.5.1           py310hff52083_0    conda-forge
matplotlib-base           3.5.1           py310h23f4a51_0    conda-forge
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
mysql-common              8.0.28               ha770c72_0    conda-forge
mysql-libs                8.0.28               hfa10184_0    conda-forge
ncurses                   6.3                  h9c3ff4c_0    conda-forge
nomkl                     1.0                  h5ca1d4c_0    conda-forge
nspr                      4.32                 h9c3ff4c_1    conda-forge
nss                       3.74                 hb5efdd6_0    conda-forge
numexpr                   2.8.0           py310hcff4476_101    conda-forge
numpy                     1.22.3          py310h45f3432_0    conda-forge
openjdk                   8.0.312              h7f98852_0    conda-forge
openjpeg                  2.4.0                hb52868f_1    conda-forge
openpyxl                  3.0.9              pyhd8ed1ab_0    conda-forge
openssl                   1.1.1l               h7f98852_0    conda-forge
packaging                 21.3               pyhd8ed1ab_0    conda-forge
pandas                    1.4.1           py310hb5077e9_0    conda-forge
parallel                  20220222             ha770c72_0    conda-forge
pcre                      8.45                 h9c3ff4c_0    conda-forge
perl                      5.32.1          2_h7f98852_perl5    conda-forge
pillow                    9.0.1           py310he619898_2    conda-forge
pip                       22.0.4             pyhd8ed1ab_0    conda-forge
pthread-stubs             0.4               h36c2ea0_1001    conda-forge
pyparsing                 3.0.7              pyhd8ed1ab_0    conda-forge
pyqt                      5.12.3          py310hff52083_8    conda-forge
pyqt-impl                 5.12.3          py310h1f8e252_8    conda-forge
pyqt5-sip                 4.19.18         py310h122e73d_8    conda-forge
pyqtchart                 5.12            py310hfcd6d55_8    conda-forge
pyqtwebengine             5.12.1          py310hfcd6d55_8    conda-forge
pytables                  3.7.0           py310hf5df6ce_0    conda-forge
python                    3.10.2          h85951f9_4_cpython    conda-forge
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python_abi                3.10                    2_cp310    conda-forge
pytz                      2021.3             pyhd8ed1ab_0    conda-forge
qt                        5.12.9               ha98a1a1_5    conda-forge
readline                  8.1                  h46c0cb4_0    conda-forge
scipy                     1.8.0           py310hea5193d_1    conda-forge
setuptools                60.9.3          py310hff52083_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
sqlite                    3.37.0               h9cd32fc_0    conda-forge
tk                        8.6.12               h27826a3_0    conda-forge
tornado                   6.1             py310h6acc77f_2    conda-forge
tzdata                    2021e                he74cb21_0    conda-forge
unicodedata2              14.0.0          py310h6acc77f_0    conda-forge
wheel                     0.37.1             pyhd8ed1ab_0    conda-forge
xorg-libxau               1.0.9                h7f98852_0    conda-forge
xorg-libxdmcp             1.1.3                h7f98852_0    conda-forge
xz                        5.2.5                h516909a_1    conda-forge
zlib                      1.2.11            h36c2ea0_1013    conda-forge
zstd                      1.5.2                ha95c52a_0    conda-forge
```

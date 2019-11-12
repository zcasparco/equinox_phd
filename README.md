CROCO
=====

Le répertoire croco contient le code CROCO de base  

Le répertoire CONFIGS contient les différentes configurations:  
- Run_MEDDY : Meddy dans une configuration idéalisée
- Run_MOZ   : Tourbillon dans la configuration réaliste du canal du Mozambique
- [Run_JETN](CONFIGS/Run_JETN/readme.md) : Jet et marée interne dans une configuration idéalisée  

Le répertoire util contient des utilitaires python  
- clean_croco.py : pour supprimer tous les xios_client* et xios_server* (sauf le 0) à partir du niveau de répertoire courant et dans les deux niveaux inférieurs  
- kill_datarmor.py : pour tuer sur datarmor tous les chainages d'une simulation en cours  
- restart_datarmor.py : pour relancer sur datarmor une simulation de plusieurs chainages  


Installation
=============

Download the repository:
```
git clone https://github.com/slgentil/croco.git
```

For pre/post processing, install an appropriate conda-environment.
Download Miniconda3 (i.e. for python3) from the [conda website](https://conda.io/miniconda.html) and run:
```
./Miniconda3-latest-Linux-x86_64.sh
conda update conda
conda create -n croco -c conda-forge python=3.7 dask dask-jobqueue \
            xarray zarr netcdf4 python-graphviz \
            jupyterlab ipywidgets \
            cartopy geopandas scikit-learn seaborn \
            hvplot geoviews datashader nodejs \
            intake-xarray gcsfs \
            cmocean gsw \
            pytide pyinterp
conda activate croco
pip install git+https://github.com/xgcm/xgcm.git
conda install pywavelets
# install croco_visu, parcels ...
cd croco; pip install -e .
jupyter labextension install @jupyter-widgets/jupyterlab-manager \
                             @pyviz/jupyterlab_pyviz \
                             jupyter-leaflet
```

see also [conda doc](doc/conda.md)

Compilation
============
Aller dans le répertoire Run_XXX  
Ce répertoire ne contient que les sources modifiés de la configuration courante par rapport à CROCO de base   
Lancer jobcomp  
L'exécutable croco et un lien symbolique vers l'exécutable xios_server.exe sont créés dans le répertoire courant

Lancement sur Datarmor
======================

```
python chain_datarmor.py workdir nbchain elaptim resolution jobname restart
```

avec:

- workdir : répertoire qui sera créé sous DATAWORK ou SCRATCH selon la variable WORK du script python
- nbchain : nombre de chainages  
- elaptim : temps elapsed pour chaque chainage HH:MM:SS  
- resolution : dépend de la configuration (4,2,1 pour JETN / 128,256,512,1024 pour MEDDY et MOZ)
- jobname : nom générique des batchs    
- restart : 0 (initial) or 1 (restart)  

Le répertoire workdir est créé dans DATAWORK ou SCRATCH  
Ce répertoire contient des sous-répertoires t1,t2,... , un répertoire par chainage  
Le script python prépare le batch et vous indique comment le lancer.  

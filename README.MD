
Commands to run the pipeline without docker
```
conda create -n ETL python=3.10 
conda activate ETL    
python DataClean.py Patient_data
python etl_from_excel.py  
python app.py  

```

Commands to run the pipeline without docker
```
docker compose up --build  

```

Open: http://127.0.0.1:8050
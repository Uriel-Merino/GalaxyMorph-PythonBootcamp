"""
Data Discovery and Retrieval
Requisit:
    Make an account in https://apps.sciserver.org/login-portal/login (SciServer)
    Documentation from SciServer: https://www.sciserver.org/docs/sciscript-python/SciServer.html
"""

# Libraries:
# import getpass
import pandas as pd
from SciServer import Authentication, CasJobs
import os

# Query and Retrieval:

def query():
    """
    Summary:
        - This function athenticates with SciServer, and joins tables with galaxy morphology
        resulting in a dataset save it as a CSV file.
    Inputs:
        - SciServer username and passowrd
    Outputs:
        - TABLE.csv a dataset from two talbes (ZooSpec galaxy morpho. classif. and PhotoObjDR7 photometry data for galaxies with morpho. classif.)
    
    """
    # SciServer Authentication: name and password is required!
    user=input("User SciServer: ")
    password=input("Password: ") 
    Authentication.login(user,password)

    # SQL query: (you can change this query to whatever query you want, but following the slides)
    TABLE="ZooSpecPhoto" # Table name
    SQL=f"""SELECT ZooSpec.*, PhotoObjDR7.* into MyDB.{TABLE}
    FROM ZooSpec INNER JOIN PhotoObjDR7
    ON PhotoObjDR7.dr7objid = ZooSpec.dr7objid"""
    # ZooSpec and PhotoObjDR7 are tables from the SDSS catalog, in our case DR7

    # Submit a job:
    jobid=CasJobs.submitJob(SQL,context="DR19") # context=database context (string)
    print(f"The query has submited: {jobid}")
    CasJobs.waitForJob(jobid,verbose=True) # Verbose is default False, but if True, will print “wait” messages on the screen while the job is still running.

    # In this next step the data has been stored in MyDB space (hint from the slide) now we can download the results in a CSV file:
    # Let's use the last hint: this output dataset has 659,272 rows (obv if you are dealing with a different dataset the size will be different)
    query_result_table=CasJobs.executeQuery(f"SELECT COUNT(*) AS n FROM {TABLE}",context="MyDB") # the query result table, in a format defined by the ‘format’ input parameter. (default pandas)
    rows=int(query_result_table.iloc[0,0]) # The output will be at 0 row and column 0
    print(f"Rows: {rows}")

    # Download the data and save it in a CSV:
    # To prevent the user from timeouts, we are gonna download the data using chunks!
    # Each chunk is written to the CSV as soon as it arrives, so nothing is lost if the script stops
    CHUNK=50000
    for i in range(0,rows,CHUNK):
        sql_mydb=(f"SELECT * FROM {TABLE} ORDER BY dr7objid "
                  f"OFFSET {i} ROWS FETCH NEXT {CHUNK} ROWS ONLY") 
        # dr7objid is organizing the data by its ID
        # offset to prevent downloading rows again
        block=CasJobs.executeQuery(sql_mydb,context="MyDB")
        block.to_csv(f"{TABLE}.csv",mode="w" if i==0 else "a",header=(i==0),index=False)
        # mode="w" + header only in the first chunk (creates the file), "a" (append) without header in the rest
        # To follow the evolution:
        print(f"Data downloaded: {min(i+CHUNK,rows)}/{rows}")
    print("All the data has been downloaded!")


def resume_download(TABLE="ZooSpecPhoto",CHUNK=50000):
    """
    Summary:
        - Continues the download of MyDB.TABLE from where TABLE.csv stopped.
        If the CSV does not exist, it starts from the beginning.
    Inputs:
        - TABLE.csv (partial) and the table in MyDB (created by query())
        - SciServer username and password
    Outputs:
        - TABLE.csv completed
    """
    user=input("User SciServer: ")
    password=input("Password: ")
    Authentication.login(user,password)

    # Total rows in the MyDB table:
    rows=int(CasJobs.executeQuery(f"SELECT COUNT(*) AS n FROM {TABLE}",context="MyDB").iloc[0,0])

    # Rows already saved in the CSV (only one column is read to be fast):
    csv=f"{TABLE}.csv"
    if os.path.exists(csv):
        start=len(pd.read_csv(csv,usecols=["dr7objid"]))
    else:
        start=0
    print(f"Rows in MyDB: {rows}, already in the CSV: {start}")

    for i in range(start,rows,CHUNK):
        sql_mydb=(f"SELECT * FROM {TABLE} ORDER BY dr7objid "
                  f"OFFSET {i} ROWS FETCH NEXT {CHUNK} ROWS ONLY")
        # dr7objid is organizing the data by its ID
        # offset to prevent downloading rows again
        block=CasJobs.executeQuery(sql_mydb,context="MyDB")
        block.to_csv(f"{TABLE}.csv",mode="w" if i==0 else "a",header=(i==0),index=False)
        # mode="w" + header only in the first chunk (creates the file), "a" (append) without header in the rest
        # To follow the evolution:
        print(f"Data downloaded: {min(i+CHUNK,rows)}/{rows}")

    # Final check:
    final=pd.read_csv(csv,usecols=["dr7objid"])
    print(f"Rows in the CSV: {len(final)} of {rows}, dr7objid unique: {final['dr7objid'].is_unique}")
    print("All the data has been downloaded!")

"""MAIN"""
if __name__ == "__main__":
    query()
    # resume_download() # If while downloading you experiment a time out!

from astroquery.sdss import SDSS

query = """
    SELECT TOP 5
        z.*,
        p.*
    FROM dbo.ZooSpec AS z
    JOIN dbo.PhotoObjDR7 AS p
    ON p.dr7objid = z.dr7objid
"""

response = SDSS.query_sql(query)
print(response[:5])
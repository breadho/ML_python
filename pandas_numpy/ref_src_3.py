

'''  1-1. importing_csv.py  '''

import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 5)

# import the land temperature data
landtemps = pd.read_csv('data/landtempssample.csv',
    names=['stationid','year','month','avgtemp','latitude',
      'longitude','elevation','station','countryid','country'],
    skiprows=1,
    parse_dates=[['month','year']],
    low_memory=False)

type(landtemps)

# show enough data to get a sense of how the import went
landtemps.head(7)
landtemps.dtypes
landtemps.shape

# fix the column name for the date
landtemps.rename(columns={'month_year':'measuredate'}, inplace=True)
landtemps.dtypes
landtemps.avgtemp.describe()
landtemps.isnull().sum()

# remove rows with missing values
landtemps.dropna(subset=['avgtemp'], inplace=True)
landtemps.shape


'''  1-1b. importing_csv_extra.py  '''

# import pandas
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 200)

# import the land temperature data
landtemps = pd.read_csv('data/landtemps.zip', compression='zip',
    names=['stationid','year','month','avgtemp','latitude',
      'longitude','elevation','station','countryid','country'],
    skiprows=1,
    parse_dates=[['month','year']],
    low_memory=False)

type(landtemps)

# show enough data to get a sense of how the import went
landtemps.head(7)
landtemps.sample(7)
landtemps.dtypes
landtemps.shape
landtemps[['avgtemp']].describe()

# remove rows with missing values
landtemps.dropna(subset=['avgtemp'], inplace=True)
landtemps.head(7)
landtemps.shape



'''  1-2. importing_excel.py  '''

# import pandas
import pandas as pd
pd.options.display.float_format = '{:.0f}'.format
pd.set_option('display.width', 85)
pd.set_option('display.max_columns', 5)

# import the land temperature data
percapitaGDP = pd.read_excel("data/GDPpercapita.xlsx",
   sheet_name="OECD.Stat export",
   skiprows=4,
   skipfooter=1,
   usecols="A,C:T")

percapitaGDP.head()
percapitaGDP.info()

# rename the Year column to metro
percapitaGDP.rename(columns={'Year':'metro'}, inplace=True)
percapitaGDP.metro.str.startswith(' ').any()
percapitaGDP.metro.str.endswith(' ').any()
percapitaGDP.metro = percapitaGDP.metro.str.strip()

# convert the data columns to numeric
for col in percapitaGDP.columns[1:]:
  percapitaGDP[col] = pd.to_numeric(percapitaGDP[col], errors='coerce')
  percapitaGDP.rename(columns={col:'pcGDP'+col}, inplace=True)

percapitaGDP.head()
percapitaGDP.dtypes
percapitaGDP.describe()

# remove rows where all of the per capita GDP values are missing
percapitaGDP.dropna(subset=percapitaGDP.columns[1:], how="all", inplace=True)
percapitaGDP.describe()
percapitaGDP.head()
percapitaGDP.shape

# set an index using the metropolitan area column
percapitaGDP.metro.count()
percapitaGDP.metro.nunique()
percapitaGDP.set_index('metro', inplace=True)
percapitaGDP.head()
percapitaGDP.loc['AUS02: Greater Melbourne']


'''  1-3. importing_sql.py  '''

# import pandas, pymssql, and mysql
import pandas as pd
import numpy as np
import pymssql
import mysql.connector
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 5)
pd.options.display.float_format = '{:,.2f}'.format

# set sql select statement to pull the data
query = "SELECT studentid, school, sex, age, famsize,\
  medu AS mothereducation, fedu AS fathereducation,\
  traveltime, studytime, failures, famrel, freetime,\
  goout, g1 AS gradeperiod1, g2 AS gradeperiod2,\
  g3 AS gradeperiod3 From studentmath"

# use the pymssql api and read_sql to retrieve and load data from a SQL Server instance
server = "pdcc.c9sqqzd5fulv.us-west-2.rds.amazonaws.com"
user = "pdccuser"
password = "pdccpass"
database = "pdcctest"
conn = pymssql.connect(server=server,
  user=user, password=password, database=database)
studentmath = pd.read_sql(query,conn)
conn.close()

# use the mysql api and read_sql to retrieve and load data from mysql
# this will result in the same file as with the pymssql 
host = "pdccmysql.c9sqqzd5fulv.us-west-2.rds.amazonaws.com"
user = "pdccuser"
password = "pdccpass"
database = "pdccschema"
connmysql = mysql.connector.connect(host=host,
  database=database,user=user,password=password)
studentmath = pd.read_sql(query,connmysql)
connmysql.close()

studentmath.dtypes
studentmath.head()

# rearrange columns and set an index
newcolorder = ['studentid', 'gradeperiod1', 'gradeperiod2',
  'gradeperiod3', 'school', 'sex', 'age', 'famsize',
  'mothereducation', 'fathereducation', 'traveltime',
  'studytime', 'freetime', 'failures', 'famrel',
  'goout']
studentmath = studentmath[newcolorder]
studentmath.studentid.count()
studentmath.studentid.nunique()
studentmath.set_index('studentid', inplace=True)
studentmath.count()

# add codes to data values
setvalues={"famrel":{1:"1:very bad",2:"2:bad",3:"3:neutral",
    4:"4:good",5:"5:excellent"},
  "freetime":{1:"1:very low",2:"2:low",3:"3:neutral",
    4:"4:high",5:"5:very high"},
  "goout":{1:"1:very low",2:"2:low",3:"3:neutral",
    4:"4:high",5:"5:very high"},
  "mothereducation":{0:np.nan,1:"1:k-4",2:"2:5-9",
    3:"3:secondary ed",4:"4:higher ed"},
  "fathereducation":{0:np.nan,1:"1:k-4",2:"2:5-9",
    3:"3:secondary ed",4:"4:higher ed"}}

studentmath.replace(setvalues, inplace=True)
setvalueskeys = [k for k in setvalues]
studentmath[setvalueskeys].memory_usage(index=False)

for col in studentmath[setvalueskeys].columns:
    studentmath[col] = studentmath[col].astype('category')

studentmath[setvalueskeys].memory_usage(index=False)

# take a closer look at the new values
studentmath['famrel'].value_counts(sort=False, normalize=True)
studentmath[['freetime','goout']].\
  apply(pd.Series.value_counts, sort=False, normalize=True)
studentmath[['mothereducation','fathereducation']].\
  apply(pd.Series.value_counts, sort=False, normalize=True)

'''  1-4. importing_spss.py  '''

# import pandas, numpy, and pyreadstat
import pandas as pd
import numpy as np
import pyreadstat
pd.set_option('display.max_columns', 5)
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 75)

# retrieve spss data, along with the meta data
nls97spss, metaspss = pyreadstat.read_sav('data/nls97.sav')
nls97spss.dtypes
nls97spss.head()

nls97spss['R0536300'].value_counts(normalize=True)

# use column labels and value labels
metaspss.variable_value_labels['R0536300']
nls97spss['R0536300'].\
  map(metaspss.variable_value_labels['R0536300']).\
  value_counts(normalize=True)
nls97spss = pyreadstat.set_value_labels(nls97spss, metaspss, formats_as_category=True)
nls97spss.columns = metaspss.column_labels
nls97spss['KEY!SEX (SYMBOL) 1997'].value_counts(normalize=True)
nls97spss.dtypes
nls97spss.columns = nls97spss.columns.\
    str.lower().\
    str.replace(' ','_').\
    str.replace('[^a-z0-9_]', '')
nls97spss.set_index('pubid__yth_id_code_1997', inplace=True)

# apply the formats from the beginning
nls97spss, metaspss = pyreadstat.read_sav('data/nls97.sav', apply_value_formats=True, formats_as_category=True)
nls97spss.columns = metaspss.column_labels
nls97spss.columns = nls97spss.columns.\
  str.lower().\
  str.replace(' ','_').\
  str.replace('[^a-z0-9_]', '')
nls97spss.dtypes
nls97spss.head()
nls97spss.govt_responsibility__provide_jobs_2006.\
  value_counts(sort=False)
nls97spss.set_index('pubid__yth_id_code_1997', inplace=True)
 
# do the same for stata data
nls97stata, metastata = pyreadstat.read_dta('data/nls97.dta', apply_value_formats=True, formats_as_category=True)
nls97stata.columns = metastata.column_labels
nls97stata.columns = nls97stata.columns.\
    str.lower().\
    str.replace(' ','_').\
    str.replace('[^a-z0-9_]', '')
nls97stata.dtypes
nls97stata.head()
nls97stata.govt_responsibility__provide_jobs_2006.\
  value_counts(sort=False)
nls97stata.min()
nls97stata.replace(list(range(-9,0)), np.nan, inplace=True)
nls97stata.min()
nls97stata.set_index('pubid__yth_id_code_1997', inplace=True)

# pull sas data, using the sas catalog file for value labels
nls97sas, metasas = pyreadstat.read_sas7bdat('data/nls97.sas7bdat', catalog_file='data/nlsformats3.sas7bcat', formats_as_category=True)
nls97sas.columns = metasas.column_labels
nls97sas.columns = nls97sas.columns.\
    str.lower().\
    str.replace(' ','_').\
    str.replace('[^a-z0-9_]', '')
nls97sas.head()
nls97sas.keysex_symbol_1997.value_counts()
nls97sas.set_index('pubid__yth_id_code_1997', inplace=True)


'''  1-5. importing_r.py  '''

# import pandas, numpy, and pyreadr
import pandas as pd
import numpy as np
import pyreadr
import pprint
pd.set_option('display.width', 72)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 25)

# get the R data
nls97r = pyreadr.read_r('data/nls97.rds')[None]
nls97r.dtypes
nls97r.head(10)

# load the value labels
with open('data/nlscodes.txt', 'r') as reader:
    setvalues = eval(reader.read())

pprint.pprint(setvalues)

newcols = ['personid','gender','birthmonth','birthyear',
  'sampletype',  'category','satverbal','satmath',
  'gpaoverall','gpaeng','gpamath','gpascience','govjobs',
  'govprices','govhealth','goveld','govind','govunemp',
  'govinc','govcollege','govhousing','govenvironment',
  'bacredits','coltype1','coltype2','coltype3','coltype4',
  'coltype5','coltype6','highestgrade','maritalstatus',
  'childnumhome','childnumaway','degreecol1',
  'degreecol2','degreecol3','degreecol4','wageincome',
  'weeklyhrscomputer','weeklyhrstv',
  'nightlyhrssleep','weeksworkedlastyear']

# set value labels, missing values, and change data type to category
nls97r.replace(setvalues, inplace=True)
nls97r.head()
nls97r.replace(list(range(-9,0)), np.nan, inplace=True)
for col in nls97r[[k for k in setvalues]].columns:
    nls97r[col] = nls97r[col].astype('category')

nls97r.dtypes

# set meaningful column headings and set category data types
nls97r.columns = newcols
nls97r.head()

'''  1-5b. importing_r.py  '''

# import pandas
import pandas as pd

# get the R data
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()
readRDS = robjects.r['readRDS']
nls97withvalues = readRDS('data/nls97withvalues.rds')

nls97withvalues


'''  1-6. persisting_tabular.py  '''

# import pandas and pyarrow
import pandas as pd
import pyarrow
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 68)
pd.set_option('display.max_columns', 3)

# import the land temperature data
landtemps = pd.read_csv('data/landtempssample.csv',
    names=['stationid','year','month','avgtemp','latitude',
      'longitude','elevation','station','countryid','country'],
    skiprows=1,
    parse_dates=[['month','year']],
    low_memory=False)

landtemps.rename(columns={'month_year':'measuredate'}, inplace=True)
landtemps.dropna(subset=['avgtemp'], inplace=True)
landtemps.dtypes
landtemps.set_index(['measuredate','stationid'], inplace=True)

# write extreme values of temperature out to Excel and CSV files
extremevals = landtemps[(landtemps.avgtemp < landtemps.avgtemp.quantile(.001)) | (landtemps.avgtemp > landtemps.avgtemp.quantile(.999))]
extremevals.shape
extremevals.sample(7)
extremevals.to_excel('views/tempext.xlsx')
extremevals.to_csv('views/tempext.csv')

# save to pickle and feather files
landtemps.to_pickle('data/landtemps.pkl')
landtemps.reset_index(inplace=True)
landtemps.to_feather("data/landtemps.ftr")

# load saved pickle and feather files
landtemps = pd.read_pickle('data/landtemps.pkl')
landtemps.head(2).T
landtemps = pd.read_feather("data/landtemps.ftr")
landtemps.head(2).T


'''  2-1. importing_json.py  '''

# import pandas, numpy, json, pprint
import pandas as pd
import numpy as np
import json
import pprint
from collections import Counter
pd.set_option('display.width', 85)
pd.set_option('display.max_columns', 8)

# load tabular structure JSON data
with open('data/allcandidatenewssample.json') as f:
  candidatenews = json.load(f)

len(candidatenews)
pprint.pprint(candidatenews[0:2])
pprint.pprint(candidatenews[0]['source'])

Counter([len(item) for item in candidatenews])
pprint.pprint(next(item for item in candidatenews if len(item)<9))
pprint.pprint(next(item for item in candidatenews if len(item)>9))
pprint.pprint([item for item in candidatenews if len(item)==2][0:10])

candidatenews = [item for item in candidatenews if len(item)>2]
len(candidatenews)

# generate counts from JSON data
politico = [item for item in candidatenews if item["source"] == "Politico"]
len(politico)
pprint.pprint(politico[0:2])
sources = [item.get('source') for item in candidatenews]
type(sources)
len(sources)
sources[0:5]
pprint.pprint(Counter(sources).most_common(10))

# fix errors in values in dictionary
for newsdict in candidatenews:
    newsdict.update((k, "The Hill") for k, v in newsdict.items()
     if k == "source" and v == "TheHill")

sources = [item.get('source') for item in candidatenews]
pprint.pprint(Counter(sources).most_common(10))

# create a pandas data frame
candidatenewsdf = pd.DataFrame(candidatenews)
candidatenewsdf.dtypes
candidatenewsdf.rename(columns={'date':'storydate'}, inplace=True)
candidatenewsdf.storydate = candidatenewsdf.storydate.astype('datetime64[ns]')
candidatenewsdf.shape
candidatenewsdf.source.value_counts(sort=True).head(10)


'''  2-2. importing_json_api.py  '''

# import pandas, numpy, json, pprint, and requests
import pandas as pd
import numpy as np
import json
import pprint
import requests

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 8)

# load more complicated data
response = requests.get("https://openaccess-api.clevelandart.org/api/artworks/?african_american_artists")
camcollections = json.loads(response.text)
print(len(camcollections['data']))
pprint.pprint(camcollections['data'][0])

# flatten the data
camcollectionsdf = pd.json_normalize(camcollections['data'], 'citations', ['accession_number','title','creation_date','collection','creators','type'])
camcollectionsdf.head(2).T

# get birth year from the creators list
creator = camcollectionsdf[:1].creators[0]
creator[0]['birth_year']
pprint.pprint(creator)
camcollectionsdf['birthyear'] = camcollectionsdf.\
  creators.apply(lambda x: x[0]['birth_year'])
camcollectionsdf.birthyear.value_counts().\
  sort_index().head()

camcollectionsdf[:1].creators[0]


'''  2-3. importing_web.py  '''

# import pandas, numpy, json, pprint, and requests
import pandas as pd
import numpy as np
import json
import pprint
import requests
from bs4 import BeautifulSoup

pd.set_option('display.width', 80)
pd.set_option('display.max_columns',6)

# parse the web page and get the header row of the table

webpage = requests.get("http://www.alrb.org/datacleaning/covidcaseoutliers.html")
bs = BeautifulSoup(webpage.text, 'html.parser')
theadrows = bs.find('table', {'id':'tblDeaths'}).thead.find_all('th')
type(theadrows)
labelcols = [j.get_text() for j in theadrows]
labelcols[0] = "rowheadings"
labelcols

# get the data from the table cells
rows = bs.find('table', {'id':'tblDeaths'}).tbody.find_all('tr')
datarows = []
labelrows = []
for row in rows:
  rowlabels = row.find('th').get_text()
  cells = row.find_all('td', {'class':'data'})
  if (len(rowlabels)>3):
    labelrows.append(rowlabels)
  if (len(cells)>0):
    cellvalues = [j.get_text() for j in cells]
    datarows.append(cellvalues)

pprint.pprint(datarows[0:2])
pprint.pprint(labelrows[0:2])

for i in range(len(datarows)):
  datarows[i].insert(0, labelrows[i])

pprint.pprint(datarows[0:2])

# load the data into pandas
totaldeaths = pd.DataFrame(datarows, columns=labelcols)
totaldeaths.iloc[:,1:5].head()
totaldeaths.dtypes
totaldeaths.columns = totaldeaths.columns.str.replace(" ", "_").str.lower()

for col in totaldeaths.columns[1:-1]:
  totaldeaths[col] = totaldeaths[col].\
    str.replace("[^0-9]","").astype('int64')

totaldeaths['hospital_beds_per_100k'] = totaldeaths['hospital_beds_per_100k'].astype('float')
totaldeaths.head()
totaldeaths.dtypes



'''  2-4. persisting_json.py  '''

# import pandas, numpy, json, pprint, and requests
import pandas as pd
import json
import pprint
import requests
import msgpack

pd.set_option('display.width', 85)
pd.set_option('display.max_columns', 8)

# load complicated JSON data from an API
response = requests.get("https://openaccess-api.clevelandart.org/api/artworks/?african_american_artists")
camcollections = json.loads(response.text)
len(camcollections['data'])
pprint.pprint(camcollections['data'][0])

# save to a json file
with open("data/camcollections.json","w") as f:
  json.dump(camcollections, f)

# read the json file
with open("data/camcollections.json","r") as f:
  camcollections = json.load(f)

pprint.pprint(camcollections['data'][0]['creators'])

# Write msgpack file
with open("data/camcollections.msgpack", "wb") as outfile:
    packed = msgpack.packb(camcollections)
    outfile.write(packed)

# Read msgpack file
with open("data/camcollections.msgpack", "rb") as data_file:
    msgbytes = data_file.read()

camcollections = msgpack.unpackb(msgbytes)

pprint.pprint(camcollections['data'][0]['creators'])



'''  3-1. firstlook.py  '''

# import pandas, numpy
import pandas as pd
import numpy as np
pd.set_option('display.width', 70)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97.csv")
covidtotals = pd.read_csv("data/covidtotals.csv",
  parse_dates=['lastdate'])

# Get basic stats on the nls dataset

nls97.set_index("personid", inplace=True)
nls97.index
nls97.shape
nls97.index.nunique()
nls97.info()
nls97.head(2).T

# Get basic stats on the covid cases dataset
covidtotals.set_index("iso_code", inplace=True)
covidtotals.index
covidtotals.shape
covidtotals.index.nunique()
covidtotals.info()
covidtotals.sample(2, random_state=1).T


'''  3-2. selecting_columns.py  '''

# import pandas and numpy, and load the nls97 data
import pandas as pd
import numpy as np
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 15)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)
nls97.loc[:, nls97.dtypes == 'object'] = \
  nls97.select_dtypes(['object']). \
  apply(lambda x: x.astype('category'))

# select a column using the pandas index operator
analysisdemo = nls97['gender']
type(analysisdemo)
analysisdemo = nls97[['gender']]
type(analysisdemo)
analysisdemo.dtypes
analysisdemo = nls97.loc[:,['gender']]
type(analysisdemo)
analysisdemo.dtypes
analysisdemo = nls97.iloc[:,[0]]
type(analysisdemo)
analysisdemo.dtypes

# select multiple columns from a pandas data frame
analysisdemo = nls97[['gender','maritalstatus',
 'highestgradecompleted']]
analysisdemo.shape
analysisdemo.head()

analysisdemo = nls97.loc[:,['gender','maritalstatus',
 'highestgradecompleted']]
analysisdemo.shape
analysisdemo.head()

# use lists to select multiple columns
keyvars = ['gender','maritalstatus',
 'highestgradecompleted','wageincome',
 'gpaoverall','weeksworked17','colenroct17']
analysiskeys = nls97[keyvars]
analysiskeys.info()

# select multiple columns using the filter operator
analysiswork = nls97.filter(like="weeksworked")
analysiswork.info()

# select multiple columns based on data types
analysiscats = nls97.select_dtypes(include=["category"])
analysiscats.info()

analysisnums = nls97.select_dtypes(include=["number"])
analysisnums.info()

# organize columns
demo = ['gender','birthmonth','birthyear']
highschoolrecord = ['satverbal','satmath','gpaoverall',
 'gpaenglish','gpamath','gpascience']
govresp = ['govprovidejobs','govpricecontrols',
  'govhealthcare','govelderliving','govindhelp',
  'govunemp','govincomediff','govcollegefinance',
  'govdecenthousing','govprotectenvironment']
demoadult = ['highestgradecompleted','maritalstatus',
  'childathome','childnotathome','wageincome',
  'weeklyhrscomputer','weeklyhrstv','nightlyhrssleep',
  'highestdegree']
weeksworked = ['weeksworked00','weeksworked01',
  'weeksworked02','weeksworked03','weeksworked04',
  'weeksworked05','weeksworked06',  'weeksworked07',
  'weeksworked08','weeksworked09','weeksworked10',
  'weeksworked11','weeksworked12','weeksworked13',
  'weeksworked14','weeksworked15','weeksworked16',
  'weeksworked17']
colenr = ['colenrfeb97','colenroct97','colenrfeb98',
  'colenroct98','colenrfeb99',  'colenroct99',
  'colenrfeb00','colenroct00','colenrfeb01',
  'colenroct01','colenrfeb02','colenroct02',
  'colenrfeb03','colenroct03','colenrfeb04',
  'colenroct04','colenrfeb05','colenroct05',
  'colenrfeb06','colenroct06','colenrfeb07',
  'colenroct07','colenrfeb08','colenroct08',
  'colenrfeb09','colenroct09','colenrfeb10',
  'colenroct10','colenrfeb11','colenroct11',
  'colenrfeb12','colenroct12','colenrfeb13',
  'colenroct13',  'colenrfeb14','colenroct14',
  'colenrfeb15','colenroct15','colenrfeb16',
  'colenroct16','colenrfeb17','colenroct17']

nls97 = nls97[demoadult + demo + highschoolrecord + \
  govresp + weeksworked + colenr]
nls97.dtypes

nls97.select_dtypes(exclude=["category"]).info()

nls97.filter(regex='income')



'''  3-3. selecting_rows.py  '''

# import pandas and numpy, and load the nls97 data
import pandas as pd
import numpy as np
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.2f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)

# use slicing to select a few rows
nls97[1000:1004].T

nls97[1000:1004:2].T

# select first 3 rows using head() and Python slicing
nls97.head(3).T
nls97[:3].T

# select last 3 rows using tail() and Python slicing
nls97.tail(3).T
nls97[-3:].T

# select a few rows using loc and iloc
nls97.loc[[195884,195891,195970]].T
nls97.loc[195884:195970].T
nls97.iloc[[0]].T
nls97.iloc[[0,1,2]].T
nls97.iloc[0:3].T
nls97.iloc[[-3,-2,-1]].T
nls97.iloc[-3:].T

# select multiple rows conditionally
nls97.nightlyhrssleep.quantile(0.05)
nls97.nightlyhrssleep.count()
sleepcheckbool = nls97.nightlyhrssleep<=4
sleepcheckbool
lowsleep = nls97.loc[sleepcheckbool]
lowsleep = nls97.loc[nls97.nightlyhrssleep<=4]
lowsleep.shape

# select rows based on multiple conditions
lowsleep.childathome.describe()
lowsleep3pluschildren = nls97.loc[(nls97.nightlyhrssleep<=4) & (nls97.childathome>=3)]
lowsleep3pluschildren.shape

# select rows based on multiple conditions and also select columns
lowsleep3pluschildren = nls97.loc[(nls97.nightlyhrssleep<=4) & (nls97.childathome>=3), ['nightlyhrssleep','childathome']]
lowsleep3pluschildren


'''  3-4. counts_categorical.py  '''

# import pandas, numpy
import pandas as pd
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.2f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)
nls97.loc[:, nls97.dtypes == 'object'] = \
  nls97.select_dtypes(['object']). \
  apply(lambda x: x.astype('category'))

# show the names of columns with category data type and check for number of missings
catcols = nls97.select_dtypes(include=["category"]).columns
nls97[catcols].isnull().sum()

# show frequencies for marital status
nls97.maritalstatus.value_counts()

# turn off sorting by frequency
nls97.maritalstatus.value_counts(sort=False)

# show percentages instead of counts
nls97.maritalstatus.value_counts(sort=False, normalize=True)

# do percentages for all government responsibility variables
nls97.filter(like="gov").apply(pd.value_counts, normalize=True)

# do percentages for all government responsibility variables for people who are married
nls97[nls97.maritalstatus=="Married"].\
filter(like="gov").\
apply(pd.value_counts, normalize=True)

# do frequencies and percentages for all category variables in data frame
freqout = open('views/frequencies.txt', 'w') 
for col in nls97.select_dtypes(include=["category"]):
  print(col, "----------------------", "frequencies",
  nls97[col].value_counts(sort=False),"percentages",
  nls97[col].value_counts(normalize=True, sort=False),
  sep="\n\n", end="\n\n\n", file=freqout)

freqout.close()


'''  3-5. stats_continuous.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.2f}'.format
covidtotals = pd.read_csv("data/covidtotals.csv",
  parse_dates=['lastdate'])
covidtotals.set_index("iso_code", inplace=True)

# look at a few rows of the covid cases data
covidtotals.shape
covidtotals.sample(2, random_state=1).T
covidtotals.dtypes

# get descriptive statistics on the cumulative values
covidtotals.describe()
totvars = ['location','total_cases','total_deaths',
  'total_cases_pm','total_deaths_pm']
covidtotals[totvars].quantile(np.arange(0.0, 1.1, 0.1))

# view the distribution of total cases
plt.hist(covidtotals['total_cases']/1000, bins=12)
plt.title("Total Covid Cases (in thousands)")
plt.xlabel('Cases')
plt.ylabel("Number of Countries")
plt.show()


'''  4-1. missing_values.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.0f}'.format
covidtotals = pd.read_csv("data/covidtotalswithmissings.csv")
covidtotals.set_index("iso_code", inplace=True)

# set up the cumulative and demographic columns
totvars = ['location','total_cases','total_deaths','total_cases_pm',
  'total_deaths_pm']
demovars = ['population','pop_density','median_age','gdp_per_capita',
  'hosp_beds']

# check the demographic columns for missing
covidtotals[demovars].isnull().sum(axis=0)
demovarsmisscnt = covidtotals[demovars].isnull().sum(axis=1)
demovarsmisscnt.value_counts()
covidtotals.loc[demovarsmisscnt>=3, ['location'] + demovars].head(5).T
type(demovarsmisscnt)

# check the cumulative columns for missing
covidtotals[totvars].isnull().sum(axis=0)
totvarsmisscnt = covidtotals[totvars].isnull().sum(axis=1)
totvarsmisscnt.value_counts()
covidtotals.loc[totvarsmisscnt>0].T

# use the fillna method to fix the mixing case data
covidtotals.total_cases_pm.fillna(covidtotals.total_cases/
  (covidtotals.population/1000000), inplace=True)
covidtotals.total_deaths_pm.fillna(covidtotals.total_deaths/
  (covidtotals.population/1000000), inplace=True)
covidtotals[totvars].isnull().sum(axis=0)



'''  4-2. outliers_univariate.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqline
import scipy.stats as scistat
pd.set_option('display.width', 85)
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.0f}'.format
covidtotals = pd.read_csv("data/covidtotals.csv")
covidtotals.set_index("iso_code", inplace=True)

# set up the cumulative and demographic columns
totvars = ['location','total_cases','total_deaths','total_cases_pm',
  'total_deaths_pm']
demovars = ['population','pop_density','median_age','gdp_per_capita',
  'hosp_beds']

# get descriptive statistics on the cumulative values
covidtotalsonly = covidtotals.loc[:, totvars]
covidtotalsonly.describe()
pd.options.display.float_format = '{:,.2f}'.format
covidtotalsonly.quantile(np.arange(0.0, 1.1, 0.1))
covidtotalsonly.skew()
covidtotalsonly.kurtosis()

# test for normality
def testnorm(var, df):
  stat, p = scistat.shapiro(df[var])
  return p

testnorm("total_cases", covidtotalsonly)
testnorm("total_deaths", covidtotalsonly)
testnorm("total_cases_pm", covidtotalsonly)
testnorm("total_deaths_pm", covidtotalsonly)

# show a qqplot for total cases and total cases per million
sm.qqplot(covidtotalsonly[['total_cases']]. \
  sort_values(['total_cases']), line='s')
plt.title("QQ Plot of Total Cases")

sm.qqplot(covidtotals[['total_cases_pm']]. \
  sort_values(['total_cases_pm']), line='s')
plt.title("QQ Plot of Total Cases Per Million")
plt.show()

# show outliers for total cases
thirdq, firstq = covidtotalsonly.total_cases.quantile(0.75), covidtotalsonly.total_cases.quantile(0.25)
interquartilerange = 1.5*(thirdq-firstq)
outlierhigh, outlierlow = interquartilerange+thirdq, firstq-interquartilerange
print(outlierlow, outlierhigh, sep=" <--> ")

# generate a table of outliers and save it to Excel
def getoutliers():
  dfout = pd.DataFrame(columns=covidtotals.columns, data=None)
  for col in covidtotalsonly.columns[1:]:
    thirdq, firstq = covidtotalsonly[col].quantile(0.75),\
      covidtotalsonly[col].quantile(0.25)
    interquartilerange = 1.5*(thirdq-firstq)
    outlierhigh, outlierlow = interquartilerange+thirdq,\
      firstq-interquartilerange
    df = covidtotals.loc[(covidtotals[col]>outlierhigh) | \
      (covidtotals[col]<outlierlow)]
    df = df.assign(varname = col, threshlow = outlierlow,\
       threshhigh = outlierhigh)
    dfout = pd.concat([dfout, df])
  return dfout

outliers = getoutliers()
outliers.varname.value_counts()
outliers.to_excel("views/outlierscases.xlsx")

# look a little more closely at outliers for cases per million
outliers.loc[outliers.varname=="total_cases_pm",\
  ['location','total_cases_pm','pop_density','gdp_per_capita']].\
  sort_values(['total_cases_pm'], ascending=False)

covidtotals[['pop_density','gdp_per_capita']].quantile([0.25,0.5,0.75])

# show the total cases histogram again
plt.hist(covidtotalsonly['total_cases']/1000, bins=7)
plt.title("Total Covid Cases (thousands)")
plt.xlabel('Cases')
plt.ylabel("Number of Countries")
plt.show()

# do a log transformation of the covid data
covidlogs = covidtotalsonly.copy()
for col in covidtotalsonly.columns[1:]:
  covidlogs[col] = np.log1p(covidlogs[col])

plt.hist(covidlogs['total_cases'], bins=7)
plt.title("Total Covid Cases (log)")
plt.xlabel('Cases')
plt.ylabel("Number of Countries")
plt.show()




'''  4-3. outliers_bivariate.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.2f}'.format
covidtotals = pd.read_csv("data/covidtotals.csv")
covidtotals.set_index("iso_code", inplace=True)

# set up the cumulative and demographic columns
totvars = ['location','total_cases','total_deaths','total_cases_pm',
  'total_deaths_pm']
demovars = ['population','pop_density','median_age','gdp_per_capita',
  'hosp_beds']

# generate a correlation matrix of the cumulative and demographic data
covidtotals.corr(method="pearson")

# get descriptive statistics on the cumulative values
covidtotalsonly = covidtotals.loc[:, totvars]

# see if some countries have unexpected low or high death rates given number of cases
covidtotalsonly['total_cases_q'] = pd.\
  qcut(covidtotalsonly['total_cases'],
  labels=['very low','low','medium',
  'high','very high'], q=5, precision=0)
covidtotalsonly['total_deaths_q'] = pd.\
  qcut(covidtotalsonly['total_deaths'],
  labels=['very low','low','medium',
  'high','very high'], q=5, precision=0)

pd.crosstab(covidtotalsonly.total_cases_q,
  covidtotalsonly.total_deaths_q)

covidtotals.loc[(covidtotalsonly.total_cases_q=="very high") & (covidtotalsonly.total_deaths_q=="medium")].T
covidtotals.loc[(covidtotalsonly.total_cases_q=="low") & (covidtotalsonly.total_deaths_q=="high")].T
covidtotals.hosp_beds.mean()

# do a scatterplot of total_cases by total_deaths
ax = sns.regplot(x="total_cases", y="total_deaths", data=covidtotals)
ax.set(xlabel="Cases", ylabel="Deaths", title="Total Covid Cases and Deaths by Country")
plt.show()

covidtotals.loc[(covidtotals.total_cases<300000) & (covidtotals.total_deaths>20000)].T
covidtotals.loc[(covidtotals.total_cases>300000) & (covidtotals.total_deaths<10000)].T

# do a scatterplot of total_cases by total_deaths
ax = sns.regplot(x="total_cases_pm", y="total_deaths_pm", data=covidtotals)
ax.set(xlabel="Cases Per Million", ylabel="Deaths Per Million", title="Total Covid Cases per Million and Deaths per Million by Country")
plt.show()

covidtotals.loc[(covidtotals.total_cases_pm<7500) \
  & (covidtotals.total_deaths_pm>250),\
  ['location','total_cases_pm','total_deaths_pm']]
covidtotals.loc[(covidtotals.total_cases_pm>5000) \
  & (covidtotals.total_deaths_pm<=50), \
  ['location','total_cases_pm','total_deaths_pm']]









'''  4-4. conditional_selection.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)

# look at some of the nls data
nls97[['wageincome','highestgradecompleted','highestdegree']].head(3).T

nls97.loc[:, "weeksworked12":"weeksworked17"].head(3).T
nls97.loc[:, "colenroct09":"colenrfeb14"].head(3).T

# show individuals with wage income but no weeks worked
nls97.loc[(nls97.weeksworked16==0) & nls97.wageincome>0, ['weeksworked16','wageincome']]

# check for ever enrolled in 4-year college
nls97.filter(like="colenr").apply(lambda x: x.str[0:1]=='3').head(2).T
nls97.filter(like="colenr").apply(lambda x: x.str[0:1]=='3').\
  any(axis=1).head(2)

# show individuals with post-graduate enrollment but no bachelor's enrollment
nobach = nls97.loc[nls97.filter(like="colenr").apply(lambda x: x.str[0:1]=='4').any(axis=1) & ~nls97.filter(like="colenr").apply(lambda x: x.str[0:1]=='3').any(axis=1), "colenrfeb97":"colenroct17"]
nobach = nls97.loc[nls97.filter(like="colenr").\
  apply(lambda x: x.str[0:1]=='4').\
  any(axis=1) & ~nls97.filter(like="colenr").\
  apply(lambda x: x.str[0:1]=='3').\
  any(axis=1), "colenrfeb97":"colenroct17"]
len(nobach)
nobach.head(3).T

# show individuals with bachelor's degrees or more but no 4-year college enrollment
nls97.highestdegree.value_counts(sort=False)
no4yearenrollment = nls97.loc[nls97.highestdegree.str[0:1].\
  isin(['4','5','6','7']) & ~nls97.filter(like="colenr").\
  apply(lambda x: x.str[0:1]=='3').\
  any(axis=1), "colenrfeb97":"colenroct17"]
len(no4yearenrollment)
no4yearenrollment.head(3).T

# show individuals with wage income more than three standard deviations greater than or less than the mean
highwages = nls97.loc[nls97.wageincome > nls97.wageincome.mean()+(nls97.wageincome.std()*3),['wageincome']]
highwages

# show individuals with large changes in weeks worked in the most recent year
workchanges = nls97.loc[~nls97.loc[:,
  "weeksworked12":"weeksworked16"].mean(axis=1).\
  between(nls97.weeksworked17*0.5,nls97.weeksworked17*2) \
  & ~nls97.weeksworked17.isnull(), 
  "weeksworked12":"weeksworked17"]
len(workchanges)
workchanges.head(7).T

# show inconsistencies between highest grade completed and highest degree
ltgrade12 = nls97.loc[nls97.highestgradecompleted<12, ['highestgradecompleted','highestdegree']]
pd.crosstab(ltgrade12.highestgradecompleted, ltgrade12.highestdegree)



'''  4-5. regression_influence.py  '''

# import pandas, numpy, matplotlib, statsmodels, and load the covid totals data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
pd.set_option('display.width', 85)
pd.options.display.float_format = '{:,.0f}'.format
covidtotals = pd.read_csv("data/covidtotals.csv")
covidtotals.set_index("iso_code", inplace=True)

# create an analysis file
xvars = ['pop_density','median_age','gdp_per_capita']

covidanalysis = covidtotals.loc[:,['total_cases_pm'] + xvars].dropna()
covidanalysis.describe()

# fit a linear regression model
def getlm(df):
  Y = df.total_cases_pm
  X = df[['pop_density','median_age','gdp_per_capita']]
  X = sm.add_constant(X)
  return sm.OLS(Y, X).fit()

lm = getlm(covidanalysis)
lm.summary()

# identify countries with an outsized influence on the model
influence = lm.get_influence().summary_frame()
influence.loc[influence.cooks_d>0.5, ['cooks_d']]
covidanalysis.loc[influence.cooks_d>0.5]

# do an influence plot
fig, ax = plt.subplots()
sm.graphics.influence_plot(lm, ax = ax, criterion="cooks")
plt.show()

# show a model without the outliers
covidanalysisminusoutliers = covidanalysis.loc[influence.cooks_d<0.5]

lm = getlm(covidanalysisminusoutliers)
lm.summary()


'''  4-6. outliers_knn.py  '''

# import pandas, pyod, and sklearn
import pandas as pd
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 20)
pd.options.display.float_format = '{:,.2f}'.format
covidtotals = pd.read_csv("data/covidtotals.csv")
covidtotals.set_index("iso_code", inplace=True)

# create a standardized dataset of the analysis variables

standardizer = StandardScaler()
analysisvars = ['location','total_cases_pm','total_deaths_pm',\
  'pop_density','median_age','gdp_per_capita']
covidanalysis = covidtotals.loc[:, analysisvars].dropna()
covidanalysisstand = standardizer.fit_transform(covidanalysis.iloc[:, 1:])

# run the KNN model and generate anomaly scores
clf_name = 'KNN'
clf = KNN(contamination=0.1)
clf.fit(covidanalysisstand)
y_pred = clf.labels_
y_scores = clf.decision_scores_

# show the predictions from the model
pred = pd.DataFrame(zip(y_pred, y_scores), 
  columns=['outlier','scores'], 
  index=covidanalysis.index)
pred.sample(10, random_state=1)
pred.outlier.value_counts()
pred.groupby(['outlier'])[['scores']].agg(['min','median','max'])

# show covid data for the outliers
covidanalysis.join(pred).loc[pred.outlier==1,\
  ['location','total_cases_pm','total_deaths_pm','scores']].\
  sort_values(['scores'], ascending=False)




'''  4-7. isolation_forest.py  '''

# import pandas, matplotlib, and scikit learn
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 80)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 7)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D
covidtotals = pd.read_csv("data/covidtotals.csv")
covidtotals.set_index("iso_code", inplace=True)

# create a standardized analysis data frame
analysisvars = ['location','total_cases_pm','total_deaths_pm',
  'pop_density','median_age','gdp_per_capita']
standardizer = StandardScaler()
covidtotals.isnull().sum()
covidanalysis = covidtotals.loc[:, analysisvars].dropna()
covidanalysisstand = standardizer.fit_transform(covidanalysis.iloc[:, 1:])

# run an isolation forest model to detect outliers
clf=IsolationForest(n_estimators=100, max_samples='auto',
  contamination=.1, max_features=1.0)
clf.fit(covidanalysisstand)
covidanalysis['anomaly'] = clf.predict(covidanalysisstand)
covidanalysis['scores'] = clf.decision_function(covidanalysisstand)
covidanalysis.anomaly.value_counts()

# view the outliers
inlier, outlier = covidanalysis.loc[covidanalysis.anomaly==1],\
  covidanalysis.loc[covidanalysis.anomaly==-1]
outlier[['location','total_cases_pm','total_deaths_pm',\
  'median_age','gdp_per_capita','scores']].\
  sort_values(['scores']).\
  head(10)

# plot the inliers and outliers
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Isolation Forest Anomaly Detection')
ax.set_zlabel("Cases Per Million")
ax.set_xlabel("GDP Per Capita")
ax.set_ylabel("Median Age")
ax.scatter3D(inlier.gdp_per_capita, inlier.median_age, inlier.total_cases_pm, label="inliers", c="blue")
ax.scatter3D(outlier.gdp_per_capita, outlier.median_age, outlier.total_cases_pm, label="outliers", c="red")
ax.legend()
plt.tight_layout()
plt.show()


'''  5-1. histograms.py  '''

# import pandas, matplotlib, and statsmodels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
landtemps = pd.read_csv("data/landtemps2019avgs.csv")
covidtotals = pd.read_csv("data/covidtotals.csv", parse_dates=["lastdate"])
covidtotals.set_index("iso_code", inplace=True)

# show some of the temperature rows 
landtemps[['station','country','latabs','elevation','avgtemp']].\
  sample(10, random_state=1)

# generate some descriptive statistics on the temperatures data
landtemps.describe()
landtemps.avgtemp.skew()
landtemps.avgtemp.kurtosis()

# plot temperature averages
plt.hist(landtemps.avgtemp)
plt.axvline(landtemps.avgtemp.mean(), color='red', linestyle='dashed', linewidth=1)
plt.title("Histogram of Average Temperatures (Celsius)")
plt.xlabel("Average Temperature")
plt.ylabel("Frequency")
plt.show()

# run a qq-plot to examine where the distribution deviates from a normal distribution
sm.qqplot(landtemps[['avgtemp']].sort_values(['avgtemp']), line='s')
plt.title("QQ Plot of Average Temperatures")
plt.show()

# show skewness and kurtosis for total_cases_pm
covidtotals.total_cases_pm.skew()
covidtotals.total_cases_pm.kurtosis()

# do a stacked histogram
showregions = ['Oceania / Aus','East Asia','Southern Africa',
  'Western Europe']

def getcases(regiondesc):
  return covidtotals.loc[covidtotals.region==regiondesc,
    'total_cases_pm']

plt.hist([getcases(k) for k in showregions],\
  color=['blue','mediumslateblue','plum','mediumvioletred'],\
  label=showregions,\
  stacked=True)
plt.title("Stacked Histogram of Cases Per Million for Selected Regions")
plt.xlabel("Cases Per Million")
plt.ylabel("Frequency")
plt.xticks(np.arange(0, 22500, step=2500))
plt.legend()
plt.show()

# show multiple histograms on one figure
fig, axes = plt.subplots(2, 2)
fig.suptitle("Histograms of Covid Cases Per Million by Selected Regions")
axes = axes.ravel()

for j, ax in enumerate(axes):
  ax.hist(covidtotals.loc[covidtotals.region==showregions[j]].\
    total_cases_pm, bins=5)
  ax.set_title(showregions[j], fontsize=10)
  for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()



'''  5-2. boxplots.py  '''

# import pandas, matplotlib, and seaborn
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)
covidtotals = pd.read_csv("data/covidtotals.csv", parse_dates=["lastdate"])
covidtotals.set_index("iso_code", inplace=True)

# do a boxplot for SAT verbal
nls97.satverbal.describe()

plt.boxplot(nls97.satverbal.dropna(), labels=['SAT Verbal'])
plt.annotate('outlier threshold', xy=(1.05,780), xytext=(1.15,780), size=7, arrowprops=dict(facecolor='black', headwidth=2, width=0.5, shrink=0.02))
plt.annotate('3rd quartile', xy=(1.08,570), xytext=(1.15,570), size=7, arrowprops=dict(facecolor='black', headwidth=2, width=0.5, shrink=0.02))
plt.annotate('median', xy=(1.08,500), xytext=(1.15,500), size=7, arrowprops=dict(facecolor='black', headwidth=2, width=0.5, shrink=0.02))
plt.annotate('1st quartile', xy=(1.08,430), xytext=(1.15,430), size=7, arrowprops=dict(facecolor='black', headwidth=2, width=0.5, shrink=0.02))
plt.annotate('outlier threshold', xy=(1.05,220), xytext=(1.15,220), size=7, arrowprops=dict(facecolor='black', headwidth=2, width=0.5, shrink=0.02))
plt.title("Boxplot of SAT Verbal Score")
plt.show()

# show some descriptives on weeks worked
weeksworked = nls97.loc[:, ['highestdegree','weeksworked16',\
  'weeksworked17']]
weeksworked.describe()

# do a box plot of weeks worked in 2016 and 2017
plt.boxplot([weeksworked.weeksworked16.dropna(),
  weeksworked.weeksworked17.dropna()],
  labels=['Weeks Worked 2016','Weeks Worked 2017'])
plt.title("Boxplots of Weeks Worked")
plt.tight_layout()
plt.show()

# show some descriptives on coronavirus cases
totvars = ['total_cases','total_deaths','total_cases_pm',\
  'total_deaths_pm']
totvarslabels = ['cases','deaths','cases per million','deaths per million']
covidtotalsonly = covidtotals[totvars]
covidtotalsonly.describe()

# do a box plot of cases and deaths per million
fig, ax = plt.subplots()
plt.title("Boxplots of Covid Cases and Deaths Per Million")
ax.boxplot([covidtotalsonly.total_cases_pm,covidtotalsonly.total_deaths_pm],\
  labels=['cases per million','deaths per million'])
plt.tight_layout()
plt.show()

# show boxplots as separate sub plots on one figure
fig, axes = plt.subplots(2, 2,)
fig.suptitle("Boxplots of Covid Cases and Deaths")
axes = axes.ravel()

for j, ax in enumerate(axes):
  ax.boxplot(covidtotalsonly.iloc[:, j], labels=[totvarslabels[j]])

plt.tight_layout()
fig.subplots_adjust(top=0.94)
plt.show()



'''  5-3. grouped_boxplots.py  '''

# import pandas, matplotlib, and seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)
covidtotals = pd.read_csv("data/covidtotals.csv", parse_dates=["lastdate"])
covidtotals.set_index("iso_code", inplace=True)

# view some descriptive statistics
def gettots(x):
  out = {}
  out['min'] = x.min()
  out['qr1'] = x.quantile(0.25)
  out['med'] = x.median()
  out['qr3'] = x.quantile(0.75)
  out['max'] = x.max()
  out['count'] = x.count()
  return pd.Series(out)

nls97.groupby(['highestdegree'])['weeksworked17'].\
  apply(gettots).unstack()

# do boxplots for weeks worked by highest degree earned
myplt = sns.boxplot('highestdegree','weeksworked17', data=nls97,
  order=sorted(nls97.highestdegree.dropna().unique()))
myplt.set_title("Boxplots of Weeks Worked by Highest Degree")
myplt.set_xlabel('Highest Degree Attained')
myplt.set_ylabel('Weeks Worked 2017')
myplt.set_xticklabels(myplt.get_xticklabels(), rotation=60, horizontalalignment='right')
plt.tight_layout()
plt.show()

# view minimum, maximum, median, and first and third quartile values
covidtotals.groupby(['region'])['total_cases_pm'].\
  apply(gettots).unstack()

# do boxplots for cases per million by region
sns.boxplot('total_cases_pm', 'region', data=covidtotals)
sns.swarmplot(y="region", x="total_cases_pm", data=covidtotals, size=2, color=".3", linewidth=0)
plt.title("Boxplots of Total Cases Per Million by Region")
plt.xlabel("Cases Per Million")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# show the most extreme value for covid totals
covidtotals.loc[covidtotals.total_cases_pm>=14000,\
  ['location','total_cases_pm']]

# do the same boxplots without the one extreme value in West Asia
sns.boxplot('total_cases_pm', 'region', data=covidtotals.loc[covidtotals.total_cases_pm<14000])
sns.swarmplot(y="region", x="total_cases_pm", data=covidtotals.loc[covidtotals.total_cases_pm<14000], size=3, color=".3", linewidth=0)
plt.title("Total Cases Without Extreme Values")
plt.xlabel("Cases Per Million")
plt.ylabel("Region")
plt.tight_layout()
plt.show()



'''  5-4. violin_plots.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)
covidtotals = pd.read_csv("data/covidtotals.csv", parse_dates=["lastdate"])

covidtotals.set_index("iso_code", inplace=True)

# do a violin plot of sat verbal scores
#sns.violinplot(nls97.satverbal, color="wheat", orient="v")
sns.violinplot(data=nls97.loc[:, ['satverbal']], color="wheat", orient="v")
plt.title("Violin Plot of SAT Verbal Score")
plt.ylabel("SAT Verbal")
plt.text(0.08, 780, 'outlier threshold', horizontalalignment='center', size='x-small')
plt.text(0.065, nls97.satverbal.quantile(0.75), '3rd quartile', horizontalalignment='center', size='x-small')
plt.text(0.05, nls97.satverbal.median(), 'Median', horizontalalignment='center', size='x-small')
plt.text(0.065, nls97.satverbal.quantile(0.25), '1st quartile', horizontalalignment='center', size='x-small')
plt.text(0.08, 210, 'outlier threshold', horizontalalignment='center', size='x-small')
plt.text(-0.4, 500, 'frequency', horizontalalignment='center', size='x-small')
plt.show()

# get some descriptives
nls97.loc[:, ['weeksworked16','weeksworked17']].describe()

# show weeks worked for 2016 and 2017
myplt = sns.violinplot(data=nls97.loc[:, ['weeksworked16','weeksworked17']])
myplt.set_title("Violin Plots of Weeks Worked")
myplt.set_xticklabels(["Weeks Worked 2016","Weeks Worked 2017"])
plt.show()

# do a violin plot of wage income by gender
nls97["maritalstatuscollapsed"] = nls97.maritalstatus.\
  replace(['Married','Never-married','Divorced','Separated','Widowed'],\
  ['Married','Never Married','Not Married','Not Married','Not Married']) 
sns.violinplot(nls97.gender, nls97.wageincome, hue=nls97.maritalstatuscollapsed, scale="count")
plt.title("Violin Plots of Wage Income by Gender and Marital Status")
plt.xlabel('Gender')
plt.ylabel('Wage Income 2017')
plt.legend(title="", loc="upper center", framealpha=0, fontsize=8)
plt.tight_layout()
plt.show()

# do a violin plot of weeks worked by degree attainment
nls97 = nls97.sort_values(['highestdegree'])
myplt = sns.violinplot('highestdegree','weeksworked17', data=nls97)
myplt.set_xticklabels(myplt.get_xticklabels(), rotation=60, horizontalalignment='right')
myplt.set_title("Violin Plots of Weeks Worked by Highest Degree")
myplt.set_xlabel('Highest Degree Attained')
myplt.set_ylabel('Weeks Worked 2017')
plt.tight_layout()
plt.show()



'''  5-5. scatter_plots.py  '''

# import pandas, matplotlib, and seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
landtemps = pd.read_csv("data/landtemps2019avgs.csv")

# run a scatter plot latitude by avgtemp
plt.scatter(x="latabs", y="avgtemp", data=landtemps)
plt.xlabel("Latitude (N or S)")
plt.ylabel("Average Temperature (Celsius)")
plt.yticks(np.arange(-60, 40, step=20))
plt.title("Latitude and Average Temperature in 2019")
plt.show()

# show the high elevation points in a different color
low, high = landtemps.loc[landtemps.elevation<=1000], landtemps.loc[landtemps.elevation>1000]
plt.scatter(x="latabs", y="avgtemp", c="blue", data=low)
plt.scatter(x="latabs", y="avgtemp", c="red", data=high)
plt.legend(('low elevation', 'high elevation'))
plt.xlabel("Latitude (N or S)")
plt.ylabel("Average Temperature (Celsius)")
plt.title("Latitude and Average Temperature in 2019")
plt.show()

# show this as a 3D plot
fig = plt.figure()
plt.suptitle("Latitude, Temperature, and Elevation in 2019")
ax.set_title('Three D')
ax = plt.axes(projection='3d')
ax.set_xlabel("Elevation")
ax.set_ylabel("Latitude")
ax.set_zlabel("Avg Temp")
ax.scatter3D(low.elevation, low.latabs, low.avgtemp, label="low elevation", c="blue")
ax.scatter3D(high.elevation, high.latabs, high.avgtemp, label="high elevation", c="red")
ax.legend()
plt.show()

# show scatter plot with a regression line
sns.regplot(x="latabs", y="avgtemp", color="blue", data=landtemps)
plt.title("Latitude and Average Temperature in 2019")
plt.xlabel("Latitude (N or S)")
plt.ylabel("Average Temperature")
plt.show()

# show scatter plot with different regression lines by elevation group
landtemps['elevation_group'] = np.where(landtemps.elevation<=1000,'low','high')
sns.lmplot(x="latabs", y="avgtemp", hue="elevation_group", palette=dict(low="blue", high="red"), legend_out=False, data=landtemps)
plt.xlabel("Latitude (N or S)")
plt.ylabel("Average Temperature")
plt.legend(('low elevation', 'high elevation'), loc='lower left')
plt.yticks(np.arange(-60, 40, step=20))
plt.title("Latitude and Average Temperature in 2019")
plt.tight_layout()
plt.show()

# check some average temperatures above the regression lines
high.loc[(high.latabs>38) & (high.avgtemp>=18),
  ['station','country','latabs','elevation','avgtemp']]
low.loc[(low.latabs>47) & (low.avgtemp>=14),
  ['station','country','latabs','elevation','avgtemp']]

# check some average temperatures below the regression lines
high.loc[(high.latabs<5) & (high.avgtemp<18),
  ['station','country','latabs','elevation','avgtemp']]
low.loc[(low.latabs<50) & (low.avgtemp<-9),
  ['station','country','latabs','elevation','avgtemp']]



'''  5-6. line_plots.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
coviddaily = pd.read_csv("data/coviddaily720.csv", parse_dates=["casedate"])

# look at a couple of sample rows of the covid daily data
coviddaily.sample(2, random_state=1).T

# calculate new cases and deaths by day
coviddailytotals = coviddaily.loc[coviddaily.casedate.\
  between('2020-02-01','2020-07-12')].\
  groupby(['casedate'])[['new_cases','new_deaths']].\
  sum().\
  reset_index()

coviddailytotals.sample(7, random_state=1)

# show line charts for new cases and new deaths by day
fig = plt.figure()
plt.suptitle("New Covid Cases and Deaths By Day Worldwide in 2020")
ax1 = plt.subplot(2,1,1)
ax1.plot(coviddailytotals.casedate, coviddailytotals.new_cases)
ax1.xaxis.set_major_formatter(DateFormatter("%b"))
ax1.set_xlabel("New Cases")
ax2 = plt.subplot(2,1,2)
ax2.plot(coviddailytotals.casedate, coviddailytotals.new_deaths)
ax2.xaxis.set_major_formatter(DateFormatter("%b"))
ax2.set_xlabel("New Deaths")
plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()

# calculate new cases and new deaths by region and day
regiontotals = coviddaily.loc[coviddaily.casedate.between('2020-02-01','2020-07-12')].\
  groupby(['casedate','region'])[['new_cases','new_deaths']].\
  sum().\
  reset_index()
regiontotals.sample(7, random_state=1)

# show plot of new cases by selected regions
showregions = ['East Asia','Southern Africa','North America',
  'Western Europe']

for j in range(len(showregions)):
  rt = regiontotals.loc[regiontotals.region==showregions[j],
    ['casedate','new_cases']]
  plt.plot(rt.casedate, rt.new_cases, label=showregions[j])

plt.title("New Covid Cases By Day and Region in 2020")
plt.gca().get_xaxis().set_major_formatter(DateFormatter("%b"))
plt.ylabel("New Cases")
plt.legend()
plt.show()

# take a closer look at the South Africa counts
af = regiontotals.loc[regiontotals.region=='Southern Africa',
  ['casedate','new_cases']].rename(columns={'new_cases':'afcases'})
sa = coviddaily.loc[coviddaily.location=='South Africa',
  ['casedate','new_cases']].rename(columns={'new_cases':'sacases'})
af = pd.merge(af, sa, left_on=['casedate'], right_on=['casedate'], how="left")
af.sacases.fillna(0, inplace=True)
af['afcasesnosa'] = af.afcases-af.sacases
afabb = af.loc[af.casedate.between('2020-04-01','2020-07-12')]

fig = plt.figure()
ax = plt.subplot()
ax.stackplot(afabb.casedate, afabb.sacases, afabb.afcasesnosa, labels=['South Africa','Other Southern Africa'])
ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
plt.title("New Covid Cases in Southern Africa")
plt.tight_layout()
plt.legend(loc="upper left")
plt.show()



'''  5-7. heat_map.py  '''

# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
covidtotals = pd.read_csv("data/covidtotals.csv", parse_dates=["lastdate"])

# generate a correlation matrix

corr = covidtotals.corr()
corr[['total_cases','total_deaths','total_cases_pm','total_deaths_pm']]

# show scatter plots
fig, axes = plt.subplots(1,2, sharey=True)
sns.regplot(covidtotals.median_age, covidtotals.total_cases_pm, ax=axes[0])
sns.regplot(covidtotals.gdp_per_capita, covidtotals.total_cases_pm, ax=axes[1])
axes[0].set_xlabel("Median Age")
axes[0].set_ylabel("Cases Per Million")
axes[1].set_xlabel("GDP Per Capita")
axes[1].set_ylabel("")
plt.suptitle("Scatter Plots of Age and GDP with Cases Per Million")
plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.show()

# generate a heat map
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="coolwarm")
plt.title('Heat Map of Correlation Matrix')
plt.tight_layout()
plt.show()



'''  6-1. series_basics.py  '''

# import pandas and load nls data
import pandas as pd
pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)

# create a series from the GPA column
gpaoverall = nls97.gpaoverall
type(gpaoverall)
gpaoverall.head()
gpaoverall.index

# select gpa values using bracket notation
gpaoverall[:5]
gpaoverall.tail()
gpaoverall[-5:]

# select values using loc
gpaoverall.loc[100061]
gpaoverall.loc[[100061]]
gpaoverall.loc[[100061,100139,100284]]
gpaoverall.loc[100061:100833]

# select values using iloc
gpaoverall.iloc[[0]]
gpaoverall.iloc[[0,1,2,3,4]]
gpaoverall.iloc[:5]
gpaoverall.iloc[-5:]



'''  6-2. summary_statistics.py  '''

# import pandas, matplotlib, and statsmodels
import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)

# show some descriptive statistics
gpaoverall = nls97.gpaoverall
gpaoverall.mean()
gpaoverall.describe()
gpaoverall.quantile(np.arange(0.1,1.1,0.1))

# subset based on values
gpaoverall.loc[gpaoverall.between(3,3.5)].head(5)
gpaoverall.loc[gpaoverall.between(3,3.5)].count()
gpaoverall.loc[(gpaoverall<2) | (gpaoverall>4)].sample(5, random_state=2)
gpaoverall.loc[gpaoverall>gpaoverall.quantile(0.99)].\
  agg(['count','min','max'])

# run tests across all values
(gpaoverall>4).any() # any person has GPA greater than 4
(gpaoverall>=0).all() # all people have GPA greater than 0
(gpaoverall>=0).sum() # of people with GPA greater than 0
(gpaoverall==0).sum() # of people with GPA equal to 0
gpaoverall.isnull().sum() # of people with missing value for GPA

# show GPA for high and low wage income earners
nls97.loc[nls97.wageincome > nls97.wageincome.quantile(0.75),'gpaoverall'].mean()
nls97.loc[nls97.wageincome < nls97.wageincome.quantile(0.25),'gpaoverall'].mean()

# show counts for series with categorical data
nls97.maritalstatus.describe()
nls97.maritalstatus.value_counts()


'''  6-3. changing_values.py  '''

# import pandas, matplotlib, and statsmodels
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)

# multiply all values of a series by a scalar
nls97.gpaoverall.head()
gpaoverall100 = nls97['gpaoverall'] * 100
gpaoverall100.head()

# use loc accessor to apply a scalar to selected rows
nls97.loc[[100061], 'gpaoverall'] = 3
nls97.loc[[100139,100284,100292],'gpaoverall'] = 0
nls97.gpaoverall.head()

# set values using more than one series
nls97['childnum'] = nls97.childathome + nls97.childnotathome
nls97.childnum.value_counts().sort_index()

# use indexing to apply a summary value to selected rows
nls97.loc[100061:100292,'gpaoverall'] = nls97.gpaoverall.mean()
nls97.gpaoverall.head()

# use iloc accessor to apply a scalar to selected rows
nls97.iloc[0, 13] = 2
nls97.iloc[1:4, 13] = 1
nls97.gpaoverall.head()

# set values after filtering
nls97.gpaoverall.nlargest()
nls97.loc[nls97.gpaoverall>4, 'gpaoverall'] = 4
nls97.gpaoverall.nlargest()

type(nls97.loc[[100061], 'gpaoverall'])
type(nls97.loc[[100061], ['gpaoverall']])


'''  6-4. changing_conditionally.py  '''

# import pandas and numpy, and load the nls and land temperatures data
import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)
landtemps = pd.read_csv("data/landtemps2019avgs.csv")

# use the numpy where function to create a categorical series with 2 values

landtemps.elevation.quantile(np.arange(0.2,1.1,0.2))

landtemps['elevation_group'] = np.where(landtemps.elevation>\
  landtemps.elevation.quantile(0.8),'High','Low')
landtemps.elevation_group = landtemps.elevation_group.astype('category')
landtemps.groupby(['elevation_group'])['elevation'].agg(['count','min','max'])

# use the numpy where function to create a categorical series with 3 values
landtemps.elevation.median()
landtemps['elevation_group'] = np.where(landtemps.elevation>
  landtemps.elevation.quantile(0.8),'High',np.where(landtemps.elevation>
  landtemps.elevation.median(),'Medium','Low'))
landtemps.elevation_group = landtemps.elevation_group.astype('category')
landtemps.groupby(['elevation_group'])['elevation'].agg(['count','min','max'])

# use numpy select to evaluate a list of conditions
test = [(nls97.gpaoverall<2) & (nls97.highestdegree=='0. None'), nls97.highestdegree=='0. None', nls97.gpaoverall<2]
result = ['1. Low GPA and No Diploma','2. No Diploma','3. Low GPA']
nls97['hsachieve'] = np.select(test, result, '4. Did Okay')
nls97[['hsachieve','gpaoverall','highestdegree']].head()
nls97.hsachieve.value_counts().sort_index()

# create a flag if individual ever had bachelor degree enrollment
nls97.loc[[100292,100583,100139], 'colenrfeb00':'colenroct04'].T
nls97['baenrollment'] = nls97.filter(like="colenr").\
  apply(lambda x: x.str[0:1]=='3').\
  any(axis=1)

nls97.loc[[100292,100583,100139], ['baenrollment']].T
nls97.baenrollment.value_counts()

# use apply and lambda to create a more complicated categorical series
def getsleepdeprivedreason(row):
  sleepdeprivedreason = "Unknown"
  if (row.nightlyhrssleep>=6):
    sleepdeprivedreason = "Not Sleep Deprived"
  elif (row.nightlyhrssleep>0):
    if (row.weeksworked16+row.weeksworked17 < 80):
      if (row.childathome>2):
        sleepdeprivedreason = "Child Rearing"
      else:
        sleepdeprivedreason = "Other Reasons"
    else:
      if (row.wageincome>=62000 or row.highestgradecompleted>=16):
        sleepdeprivedreason = "Work Pressure"
      else:
        sleepdeprivedreason = "Income Pressure"
  else:
    sleepdeprivedreason = "Unknown"
  return sleepdeprivedreason

nls97['sleepdeprivedreason'] = nls97.apply(getsleepdeprivedreason, axis=1)
nls97.sleepdeprivedreason = nls97.sleepdeprivedreason.astype('category')
nls97.sleepdeprivedreason.value_counts()


'''  6-5. series_strings.py  '''

# import pandas and numpy, and load the nls and data
import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
nls97 = pd.read_csv("data/nls97c.csv")
nls97.set_index("personid", inplace=True)

# tests whether a string pattern exists in a string
nls97.govprovidejobs.value_counts()
nls97['govprovidejobsdefprob'] = np.where(nls97.govprovidejobs.isnull(),
  np.nan,np.where(nls97.govprovidejobs.str.contains("not"),"No","Yes"))
pd.crosstab(nls97.govprovidejobs, nls97.govprovidejobsdefprob)

# handle leading or trailing spaces in a string
nls97.maritalstatus.value_counts()
nls97.maritalstatus.str.startswith(' ').any()
nls97.maritalstatus.str.endswith(' ').any()
nls97['evermarried'] = np.where(nls97.maritalstatus.isnull(),np.nan,np.where(nls97.maritalstatus.str.strip()=="Never-married","No","Yes"))
pd.crosstab(nls97.maritalstatus, nls97.evermarried)

# use isin to compare a string value to a list of values
nls97['receivedba'] = np.where(nls97.highestdegree.isnull(),np.nan,np.where(nls97.highestdegree.str[0:1].isin(['4','5','6','7']),"Yes","No"))
pd.crosstab(nls97.highestdegree, nls97.receivedba)

# convert a text response to numeric using numbers in the text
pd.concat([nls97.weeklyhrstv.head(),\
  nls97.weeklyhrstv.str.findall("\d+").head()], axis=1)

def getnum(numlist):
  highval = 0
  if (type(numlist) is list):
    lastval = int(numlist[-1])
    if (numlist[0]=='40'):
      highval = 45
    elif (lastval==2):
      highval = 1
    else:
      highval = lastval - 5
  else:
    highval = np.nan
  return highval

nls97['weeklyhrstvnum'] = nls97.weeklyhrstv.str.\
  findall("\d+").apply(getnum)
pd.crosstab(nls97.weeklyhrstv, nls97.weeklyhrstvnum)


# replace values in a series with alternative values
comphrsold = ['None','Less than 1 hour a week',
  '1 to 3 hours a week','4 to 6 hours a week',
  '7 to 9 hours a week','10 hours or more a week']
comphrsnew = ['A. None','B. Less than 1 hour a week',
  'C. 1 to 3 hours a week','D. 4 to 6 hours a week',
  'E. 7 to 9 hours a week','F. 10 hours or more a week']
nls97.weeklyhrscomputer.value_counts().sort_index()
nls97.weeklyhrscomputer.replace(comphrsold, comphrsnew, inplace=True)
nls97.weeklyhrscomputer.value_counts().sort_index()








'''  6-6. date_transform.py  '''

# import pandas
import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 220)
pd.options.display.float_format = '{:,.0f}'.format
covidcases = pd.read_csv("data/covidcases720.csv")
nls97 = pd.read_csv("data/nls97c.csv")
nls97.set_index("personid", inplace=True)

# show the birth month and year values
nls97[['birthmonth','birthyear']].head()
nls97[['birthmonth','birthyear']].isnull().sum()
nls97.birthmonth.value_counts().sort_index()
nls97.birthyear.value_counts().sort_index()

# use fillna to fix missing value
nls97.birthmonth.fillna(int(nls97.birthmonth.mean()), inplace=True)
nls97.birthmonth.value_counts().sort_index()

# use month and date integers to create a datetime column
nls97['birthdate'] = pd.to_datetime(dict(year=nls97.birthyear, month=nls97.birthmonth, day=15))
nls97.birthdate.describe()
nls97[['birthmonth','birthyear','birthdate']].head()
nls97[['birthmonth','birthyear','birthdate']].isnull().sum()

# define a function for calculating given start and end date
def calcage(startdate, enddate):
  age = enddate.year - startdate.year
  if (enddate.month<startdate.month or (enddate.month==startdate.month and enddate.day<startdate.day)):
    age = age -1
  return age

# calculate age
rundate = pd.to_datetime('2020-07-20')
nls97["age"] = nls97.apply(lambda x: calcage(x.birthdate, rundate), axis=1)
nls97.loc[100061:100583, ['age','birthdate']]

# convert a string column to a datetime column
covidcases.iloc[:, 0:6].dtypes
covidcases.iloc[:, 0:6].sample(2, random_state=1).T
covidcases['casedate'] = pd.to_datetime(covidcases.casedate, format='%Y-%m-%d')
covidcases.iloc[:, 0:6].dtypes

# get descriptive statistics on datetime column
covidcases.casedate.describe()

# calculate days since first case by country
firstcase = covidcases.loc[covidcases.new_cases>0,['location','casedate']].\
  sort_values(['location','casedate']).\
  drop_duplicates(['location'], keep='first').\
  rename(columns={'casedate':'firstcasedate'})
covidcases = pd.merge(covidcases, firstcase, left_on=['location'], right_on=['location'], how="left")
covidcases['dayssincefirstcase'] = covidcases.casedate - covidcases.firstcasedate
covidcases.dayssincefirstcase.describe()







'''  6-7. cleaning_missings.py  '''

# import pandas
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97c.csv")
nls97.set_index("personid", inplace=True)

# set up school record and demographic data frames from the NLS data
schoolrecordlist = ['satverbal','satmath','gpaoverall','gpaenglish',
  'gpamath','gpascience','highestdegree','highestgradecompleted']

demolist = ['maritalstatus','childathome','childnotathome',
  'wageincome','weeklyhrscomputer','weeklyhrstv','nightlyhrssleep']
schoolrecord = nls97[schoolrecordlist]

demo = nls97[demolist]
schoolrecord.shape
demo.shape

# check the school record data for missings
schoolrecord.isnull().sum(axis=0)
misscnt = schoolrecord.isnull().sum(axis=1)
misscnt.value_counts().sort_index()
schoolrecord.loc[misscnt>=7].head(4).T

# remove rows with almost all missing data
schoolrecord = schoolrecord.dropna(thresh=2)
schoolrecord.shape
schoolrecord.isnull().sum(axis=1).value_counts().sort_index()

# assign mean values to missings
int(schoolrecord.gpaoverall.mean())
schoolrecord.gpaoverall.isnull().sum()
schoolrecord.gpaoverall.fillna(int(schoolrecord.gpaoverall.mean()), inplace=True)
schoolrecord.gpaoverall.isnull().sum()

# use forward fill
demo.wageincome.head().T
demo.wageincome.isnull().sum()
nls97.wageincome.fillna(method='ffill', inplace=True)
demo = nls97[demolist]
demo.wageincome.head().T
demo.wageincome.isnull().sum()

# fill missings with the average by group
nls97[['highestdegree','weeksworked17']].head()
workbydegree = nls97.groupby(['highestdegree'])['weeksworked17'].mean().\
  reset_index().rename(columns={'weeksworked17':'meanweeksworked17'})
nls97 = nls97.reset_index().\
  merge(workbydegree, left_on=['highestdegree'], right_on=['highestdegree'], how='left').set_index('personid')
nls97.weeksworked17.fillna(nls97.meanweeksworked17, inplace=True)
nls97[['highestdegree','weeksworked17','meanweeksworked17']].head()


'''  6-8. impute_missings_knn.py  '''

# import pandas and scikit learn's KNNImputer module
import pandas as pd
from sklearn.impute import KNNImputer
pd.options.display.float_format = '{:,.1f}'.format
nls97 = pd.read_csv("data/nls97c.csv")
nls97.set_index("personid", inplace=True)

# load the NLS school record data
schoolrecordlist = ['satverbal','satmath','gpaoverall','gpaenglish',
  'gpamath','gpascience','highestgradecompleted']
schoolrecord = nls97[schoolrecordlist]

# initialize a KNN imputation model and fill values
impKNN = KNNImputer(n_neighbors=5)
newvalues = impKNN.fit_transform(schoolrecord)
schoolrecordimp = pd.DataFrame(newvalues, columns=schoolrecordlist, index=schoolrecord.index)

# view imputed values
schoolrecord.head().T
schoolrecordimp.head().T
schoolrecord[['gpaoverall','highestgradecompleted']].agg(['mean','count'])
schoolrecordimp[['gpaoverall','highestgradecompleted']].agg(['mean','count'])


'''  7-1. row_iteration.py  '''

# import pandas and numpy, and load the covid data
import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
coviddaily = pd.read_csv("data/coviddaily720.csv", parse_dates=["casedate"])
ltbrazil = pd.read_csv("data/ltbrazil.csv")

# sort the covid data by location and case date in ascending order
coviddaily = coviddaily.sort_values(['location','casedate'])

# iterate over rows with itertuples, append to list with each change of group
prevloc = 'ZZZ'
rowlist = []
for row in coviddaily.itertuples():
  if (prevloc!=row.location):
    if (prevloc!='ZZZ'):
      rowlist.append({'location':prevloc, 'casecnt':casecnt})
    casecnt = 0
    prevloc = row.location
  casecnt += row.new_cases

rowlist.append({'location':prevloc, 'casecnt':casecnt})
len(rowlist)
rowlist[0:4]

# create a dataframe from the rowlist
covidtotals = pd.DataFrame(rowlist)
covidtotals.head()

# sort the land temperatures data and drop rows with missing values for temperature
ltbrazil = ltbrazil.sort_values(['station','month'])
ltbrazil = ltbrazil.dropna(subset=['temperature'])

# iterate over rows with itertuples, append to list with each change of group
prevstation = 'ZZZ'
prevtemp = 0
rowlist = []

for row in ltbrazil.itertuples():
  if (prevstation!=row.station):
    if (prevstation!='ZZZ'):
      rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt,
                      'stationcnt':stationcnt})
    tempcnt = 0
    stationcnt = 0
    prevstation = row.station
  # choose only rows that are within 3 degrees of the previous temperature  
  if ((0 <= abs(row.temperature-prevtemp) <= 3) or (stationcnt==0)):
    tempcnt += row.temperature
    stationcnt += 1
  prevtemp = row.temperature

rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt, 'stationcnt':stationcnt})
rowlist[0:5]
ltbrazilavgs = pd.DataFrame(rowlist)
ltbrazilavgs.head()




'''  7-2. numpy_iteration.py  '''

# import pandas and numpy, and load the covid data
import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
coviddaily = pd.read_csv("data/coviddaily720.csv", parse_dates=["casedate"])
ltbrazil = pd.read_csv("data/ltbrazil.csv")

# create a list of locations
loclist = coviddaily.location.unique().tolist()

# use a numpy array to calculate sums
rowlist = []
casevalues = coviddaily[['location','new_cases']].to_numpy()
for locitem in loclist:
  cases = [casevalues[j][1] for j in range(len(casevalues))\
    if casevalues[j][0]==locitem]
  rowlist.append(sum(cases))

len(rowlist)
len(loclist)
rowlist[0:5]
casetotals = pd.DataFrame(zip(loclist,rowlist), columns=(['location','casetotals']))
casetotals.head()

# sort the land temperatures data and drop rows with missing values for temperature
ltbrazil = ltbrazil.sort_values(['station','month'])
ltbrazil = ltbrazil.dropna(subset=['temperature'])

# iterate using numpy arrays
prevstation = 'ZZZ'
prevtemp = 0
rowlist = []
tempvalues = ltbrazil[['station','temperature']].to_numpy()
for j in range(len(tempvalues)):
  station = tempvalues[j][0]
  temperature = tempvalues[j][1]
  if (prevstation!=station):
    if (prevstation!='ZZZ'):
      rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt, 'stationcnt':stationcnt})
    tempcnt = 0
    stationcnt = 0
    prevstation = station
  
  if ((0 <= abs(temperature-prevtemp) <= 3) or (stationcnt==0)):
    tempcnt += temperature
    stationcnt += 1
  
  prevtemp = temperature

rowlist.append({'station':prevstation, 'avgtemp':tempcnt/stationcnt, 'stationcnt':stationcnt})
rowlist[0:5]

# create a data frame of land temperature averages
ltbrazilavgs = pd.DataFrame(rowlist)
ltbrazilavgs.head()


'''  7-3. groupby_basics.py  '''

# import pandas and numpy, and load the covid data
import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 50)
pd.options.display.float_format = '{:,.0f}'.format
coviddaily = pd.read_csv("data/coviddaily720.csv", parse_dates=["casedate"])

# create a pandas groupby data frame
countrytots = coviddaily.groupby(['location'])
type(countrytots)

# create data frames for the first and last rows for each country
countrytots.first().iloc[0:5, 0:5]
countrytots.last().iloc[0:5, 0:5]
type(countrytots.last())

# get all of the rows for a country
countrytots.get_group('Zimbabwe').iloc[0:5, 0:5]

# loop through the groups
for name, group in countrytots:
  if (name in ['Malta','Kuwait']):
    print(group.iloc[0:5, 0:5])

# show the number of rows for each country
countrytots.size()

# show summary statistics by country
countrytots.new_cases.describe().head()
countrytots.new_cases.sum().head()



'''  7-4. groupby_more.py  '''

# import pandas, load the nls97 feather file
import pandas as pd
pd.set_option('display.width', 90)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 30)
pd.options.display.float_format = '{:,.1f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)

# review the structure of the nls97 data
nls97.iloc[:,0:7].info()

# look again at some of the data
catvars = ['gender','maritalstatus','highestdegree']

for col in catvars:
  print(col, nls97[col].value_counts().sort_index(), sep="\n\n", end="\n\n\n")


# review some descriptive statistics
contvars = ['satmath','satverbal','weeksworked06','gpaoverall',
  'childathome']

nls97[contvars].describe()

# look at sat math scores by gender
nls97.groupby('gender')['satmath'].mean()

# look at sat math scores by gender and highest degree earned
nls97.groupby(['gender','highestdegree'])['satmath'].mean()

# look at sat math and verbal scores by gender and highest degree earned
nls97.groupby(['gender','highestdegree'])[['satmath','satverbal']].mean()

# add max and standard deviations
nls97.groupby(['gender','highestdegree'])['gpaoverall'].agg(['count','mean','max','std'])

# use a dictionary for more complicated aggregations
pd.options.display.float_format = '{:,.1f}'.format
aggdict = {'weeksworked06':['count', 'mean', 'max','std'], 'childathome':['count', 'mean', 'max', 'std']}
nls97.groupby(['highestdegree']).agg(aggdict)
nls97.groupby(['maritalstatus']).agg(aggdict)


'''  7-5. groupby_udf.py  '''

# import pandas and numpy, and load the nls data
import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.1f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)

# create a function for calculating interquartile range
def iqr(x):
  return x.quantile(0.75) - x.quantile(0.25)

# run the interquartile range function
aggdict = {'weeksworked06':['count', 'mean', iqr], 'childathome':['count', 'mean', iqr]}
nls97.groupby(['highestdegree']).agg(aggdict)

# define a function to return the summary statistics as a series
def gettots(x):
  out = {}
  out['qr1'] = x.quantile(0.25)
  out['med'] = x.median()
  out['qr3'] = x.quantile(0.75)
  out['count'] = x.count()
  return pd.Series(out)

# use apply to run the function
pd.options.display.float_format = '{:,.0f}'.format
nls97.groupby(['highestdegree'])['weeksworked06'].apply(gettots)

# chain reset_index to set the default index
nls97.groupby(['highestdegree'])['weeksworked06'].apply(gettots).reset_index()

# allow the index to be created
nlssums = nls97.groupby(['highestdegree'])['weeksworked06'].apply(gettots).unstack()
nlssums
nlssums.info()

# run the groupby without creating an index and create a data frame
nls97.groupby(['highestdegree'], as_index=False)['weeksworked06'].apply(gettots)


'''  7-6. groupby_to_dataframe.py  '''

# import pandas and load the covid data and land temperature data
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 50)
pd.options.display.float_format = '{:,.0f}'.format
coviddaily = pd.read_csv("data/coviddaily720.csv", parse_dates=["casedate"])
ltbrazil = pd.read_csv("data/ltbrazil.csv")

# convert covid data from one country per day to summary values across all countries per day
coviddailytotals = coviddaily.loc[coviddaily.casedate.\
  between('2020-02-01','2020-07-12')].\
  groupby(['casedate'], as_index=False)[['new_cases','new_deaths']].\
  sum()

coviddailytotals.head(10)

# create a data frame with average temperatures from each station in Brazil
ltbrazil = ltbrazil.dropna(subset=['temperature'])
#ltbrazil.loc[103508:104551, ['station','year','month','temperature','elevation','latabs']]
ltbrazil[['station','year','month','temperature','elevation','latabs']]
ltbrazilavgs = ltbrazil.groupby(['station'], as_index=False).\
  agg({'latabs':'first','elevation':'first','temperature':'mean'})
ltbrazilavgs.head(10)



'''  8-1. combining_vertically.py  '''

# import pandas and numpy
import pandas as pd
import numpy as np
import os
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 50)
pd.options.display.float_format = '{:,.0f}'.format

# load the data for Cameroon and Poland
ltcameroon = pd.read_csv("data/ltcountry/ltcameroon.csv")
ltpoland = pd.read_csv("data/ltcountry/ltpoland.csv")

# concatenate the Cameroon and Poland data
ltcameroon.shape
ltpoland.shape
ltall = pd.concat([ltcameroon, ltpoland])
ltall.country.value_counts()

# concatenate all of the data files
directory = "data/ltcountry"
ltall = pd.DataFrame()
for filename in os.listdir(directory):
  if filename.endswith(".csv"): 
    fileloc = os.path.join(directory, filename)

    # open the next file
    with open(fileloc) as f:
      ltnew = pd.read_csv(fileloc)
      print(filename + " has " + str(ltnew.shape[0]) + " rows.")
      ltall = pd.concat([ltall, ltnew])

      # check for differences in columns
      columndiff = ltall.columns.symmetric_difference(ltnew.columns)
      if (not columndiff.empty):
        print("", "Different column names for:", filename,\
          columndiff, "", sep="\n")


ltall[['country','station','month','temperature','latitude']].sample(5, random_state=1)

# check values in the concatenated data
ltall.country.value_counts().sort_index()
ltall.groupby(['country']).agg({'temperature':['min','mean',\
  'max','count'],'latabs':['min','mean','max','count']})

# fix missing values
ltall['latabs'] = np.where(ltall.country=="Oman", ltall.latitude, ltall.latabs)
ltall.groupby(['country']).agg({'temperature':['min','mean',\
  'max','count'],'latabs':['min','mean','max','count']})


'''  8-2. one_to_one_merge.py  '''

# import pandas and numpy, and load the nls data
import pandas as pd
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97f.csv")
nls97.set_index("personid", inplace=True)
nls97add = pd.read_csv("data/nls97add.csv")

# look at some of the nls data
nls97.head()
nls97.shape
nls97add.head()
nls97add.shape

# check for unique ids
nls97.originalid.nunique()==nls97.shape[0]
nls97add.originalid.nunique()==nls97add.shape[0]

# create some mismatched ids
nls97 = nls97.sort_values('originalid')
nls97add = nls97add.sort_values('originalid')
nls97.iloc[0:2, -1] = nls97.originalid+10000
nls97.originalid.head(2)
nls97add.iloc[0:2, 0] = nls97add.originalid+20000
nls97add.originalid.head(2)

# use join to do a left join
nlsnew = nls97.join(nls97add.set_index(['originalid']))
nlsnew.loc[nlsnew.originalid>9999, ['originalid','gender','birthyear','motherage','parentincome']]

# do a left join with merge
nlsnew = pd.merge(nls97, nls97add, on=['originalid'], how="left")
nlsnew.loc[nlsnew.originalid>9999, ['originalid','gender','birthyear','motherage','parentincome']]

# do a right join
nlsnew = pd.merge(nls97, nls97add, on=['originalid'], how="right")
nlsnew.loc[nlsnew.originalid>9999, ['originalid','gender','birthyear','motherage','parentincome']]

# do an inner join
nlsnew = pd.merge(nls97, nls97add, on=['originalid'], how="inner")
nlsnew.loc[nlsnew.originalid>9999, ['originalid','gender','birthyear','motherage','parentincome']]

# do an outer join
nlsnew = pd.merge(nls97, nls97add, on=['originalid'], how="outer")
nlsnew.loc[nlsnew.originalid>9999, ['originalid','gender','birthyear','motherage','parentincome']]

# create a function to check id mismatches
def checkmerge(dfleft, dfright, idvar):
  dfleft['inleft'] = "Y"
  dfright['inright'] = "Y"
  dfboth = pd.merge(dfleft[[idvar,'inleft']],\
    dfright[[idvar,'inright']], on=[idvar], how="outer")
  dfboth.fillna('N', inplace=True)
  print(pd.crosstab(dfboth.inleft, dfboth.inright))

checkmerge(nls97,nls97add, "originalid")



nlsnew = pd.merge(nls97, nls97add, left_on=['originalid'], right_on=['originalid'], how="right")
nlsnew.loc[nlsnew.originalid>9999, ['originalid','gender','birthyear','motherage','parentincome']]


'''  8-3. multiple_columns.py  '''

# import pandas, and load the nls weeks worked and college data
import pandas as pd
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
nls97weeksworked = pd.read_csv("data/nls97weeksworked.csv")
nls97colenr = pd.read_csv("data/nls97colenr.csv")

# look at some of the nls data
nls97weeksworked.sample(10, random_state=1)
nls97weeksworked.shape

nls97weeksworked.originalid.nunique()
nls97colenr.sample(10, random_state=1)
nls97colenr.shape
nls97colenr.originalid.nunique()

# check for unique ids
nls97weeksworked.groupby(['originalid','year'])\
  ['originalid'].count().shape
nls97colenr.groupby(['originalid','year'])\
  ['originalid'].count().shape

# create a function to check id mismatches
def checkmerge(dfleft, dfright, idvar):
  dfleft['inleft'] = "Y"
  dfright['inright'] = "Y"
  dfboth = pd.merge(dfleft[idvar + ['inleft']],\
    dfright[idvar + ['inright']], on=idvar, how="outer")
  dfboth.fillna('N', inplace=True)
  print(pd.crosstab(dfboth.inleft, dfboth.inright))

checkmerge(nls97weeksworked.copy(),nls97colenr.copy(), ['originalid','year'])

# use multiple merge-by columns
nlsworkschool = pd.merge(nls97weeksworked, nls97colenr, on=['originalid','year'], how="inner")
nlsworkschool.shape
nlsworkschool.sample(10, random_state=1)

'''  8-4. one_to_many_merge.py  '''

# import pandas and the land temperatures data
import pandas as pd
countries = pd.read_csv("data/ltcountries.csv")
locations = pd.read_csv("data/ltlocations.csv")

# set index for the locations and countries data and print a few rows
countries.set_index(['countryid'], inplace=True)
locations.set_index(['countryid'], inplace=True)
countries.head()
countries.index.nunique()==countries.shape[0]
locations[['locationid','latitude','stnelev']].head(10)

# do a left join of countries to locations 
stations = countries.join(locations)
stations[['locationid','latitude','stnelev','country']].head(10)

# reload the locations file and check the merge
countries = pd.read_csv("data/ltcountries.csv")
locations = pd.read_csv("data/ltlocations.csv")
def checkmerge(dfleft, dfright, idvar):
  dfleft['inleft'] = "Y"
  dfright['inright'] = "Y"
  dfboth = pd.merge(dfleft[[idvar,'inleft']],\
    dfright[[idvar,'inright']], on=[idvar], how="outer")
  dfboth.fillna('N', inplace=True)
  print(pd.crosstab(dfboth.inleft, dfboth.inright))
  print(dfboth.loc[(dfboth.inleft=='N') | (dfboth.inright=='N')])

checkmerge(countries.copy(), locations.copy(), "countryid")

# show rows in one file and not another
countries.loc[countries.countryid.isin(["LQ","ST"])]
locations.loc[locations.countryid=="FO"]

# merge location and country data
stations = pd.merge(countries, locations, on=["countryid"], how="left")
stations[['locationid','latitude','stnelev','country']].head(10)
stations.shape
stations.loc[stations.countryid.isin(["LQ","ST"])].isnull().sum()



'''  8-5. many_to_many_merge.py  '''

# import pandas and the CMA collections data
import pandas as pd
cmacitations = pd.read_csv("data/cmacitations.csv")
cmacreators = pd.read_csv("data/cmacreators.csv")

# look at the citations data
cmacitations.head(10)
cmacitations.shape
cmacitations.id.nunique()

# look at the creators data
cmacreators.loc[:,['id','creator','birth_year']].head(10)
cmacreators.shape
cmacreators.id.nunique()

# show duplications of merge-by values for citations
cmacitations.id.value_counts().head(10)

# show duplications of merge-by values for creators
cmacreators.id.value_counts().head(10)

# check the merge
def checkmerge(dfleft, dfright, idvar):
  dfleft['inleft'] = "Y"
  dfright['inright'] = "Y"
  dfboth = pd.merge(dfleft[[idvar,'inleft']],\
    dfright[[idvar,'inright']], on=[idvar], how="outer")
  dfboth.fillna('N', inplace=True)
  print(pd.crosstab(dfboth.inleft, dfboth.inright))

checkmerge(cmacitations.copy(), cmacreators.copy(), "id")

# show a merge-by column duplicated in both data frames
cmacitations.loc[cmacitations.id==124733]
cmacreators.loc[cmacreators.id==124733, ['id','creator','birth_year','title']]

# do a many-to-many merge
cma = pd.merge(cmacitations, cmacreators, on=['id'], how="outer")
cma['citation'] = cma.citation.str[0:20]
cma['creator'] = cma.creator.str[0:20]
cma.loc[cma.id==124733, ['citation','creator','birth_year']]



'''  8-6. merge_routine.py  '''

# import pandas and the land temperatures data
import pandas as pd
countries = pd.read_csv("data/ltcountries.csv")
locations = pd.read_csv("data/ltlocations.csv")

# check the merge-by column matches
def checkmerge(dfleft, dfright, mergebyleft, mergebyright):
  dfleft['inleft'] = "Y"
  dfright['inright'] = "Y"
  dfboth = pd.merge(dfleft[[mergebyleft,'inleft']],\
    dfright[[mergebyright,'inright']], left_on=[mergebyleft],\
    right_on=[mergebyright], how="outer")
  dfboth.fillna('N', inplace=True)
  print(pd.crosstab(dfboth.inleft, dfboth.inright))
  print(dfboth.loc[(dfboth.inleft=='N') | (dfboth.inright=='N')].head(20))

checkmerge(countries.copy(), locations.copy(), "countryid", "countryid")

# merge location and country data
stations = pd.merge(countries, locations, left_on=["countryid"], right_on=["countryid"], how="left")
stations[['locationid','latitude','stnelev','country']].head(10)
stations.shape


'''  9-a1. remove_duplicates.py  '''

# import pandas, numpy, and covid cases daily data
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format
covidcases = pd.read_csv("data/covidcases720.csv")

# create lists for daily cases, for cumulative columns and for demographic columns
dailyvars = ['casedate','new_cases','new_deaths']
totvars = ['location','total_cases','total_deaths']

demovars = ['population','population_density','median_age',
  'gdp_per_capita','hospital_beds_per_thousand','region']
covidcases[dailyvars + totvars + demovars].head(3).T

# create a daily data frames
coviddaily = covidcases[['location'] + dailyvars]
coviddaily.shape
coviddaily.head()

# select one row per country
covidcases.location.nunique()
coviddemo = covidcases[['casedate'] + totvars + demovars].\
  sort_values(['location','casedate']).\
  drop_duplicates(['location'], keep='last').\
  rename(columns={'casedate':'lastdate'})

coviddemo.shape
coviddemo.head(3).T

# sum values for each group
covidtotals = covidcases.groupby(['location'], as_index=False).\
  agg({'new_cases':'sum','new_deaths':'sum','median_age':'last',
    'gdp_per_capita':'last','region':'last','casedate':'last',
    'population':'last'}).\
  rename(columns={'new_cases':'total_cases',
    'new_deaths':'total_deaths','casedate':'lastdate'})
  
covidtotals.head(3).T


'''  9-a2. many_to_many_reshape.py  '''

# import pandas and the CMA collections data
import pandas as pd
pd.options.display.float_format = '{:,.0f}'.format
cma = pd.read_csv("data/cmacollections.csv")

# show the cma collections data
cma.shape
cma.head(2).T
cma.id.nunique()
cma.drop_duplicates(['id','citation']).id.count()
cma.drop_duplicates(['id','creator']).id.count()

# show a collection item with duplicated citations and creators
cma.set_index(['id'], inplace=True)
cma.loc[124733, ['title','citation','creator','birth_year']].head(14)

# create a collections data frame
collectionsvars = ['title','collection','type']
cmacollections = cma[collectionsvars].\
  reset_index().\
  drop_duplicates(['id']).\
  set_index(['id'])
cmacollections.shape
cmacollections.head()
cmacollections.loc[124733]

# create a citations data frame
cmacitations = cma[['citation']].\
  reset_index().\
  drop_duplicates(['id','citation']).\
  set_index(['id'])
cmacitations.loc[124733]

# create a creators data frame
creatorsvars = ['creator','birth_year','death_year']
cmacreators = cma[creatorsvars].\
  reset_index().\
  drop_duplicates(['id','creator']).\
  set_index(['id'])
cmacreators.loc[124733]

# count the number of collection items with a creator born after 1950
cmacreators['birth_year'] = cmacreators.birth_year.str.findall("\d+").str[0].astype(float)
youngartists = cmacreators.loc[cmacreators.birth_year>1950, ['creator']].assign(creatorbornafter1950='Y')
youngartists.shape[0]==youngartists.index.nunique()
youngartists
cmacollections = pd.merge(cmacollections, youngartists, left_on=['id'], right_on=['id'], how='left')
cmacollections.creatorbornafter1950.fillna("N", inplace=True)
cmacollections.shape
cmacollections.creatorbornafter1950.value_counts()


'''  9-a3. stack_melt.py  '''

# import pandas and load the nls data
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97f.csv")

# view some of the weeks worked values
nls97.set_index(['originalid'], inplace=True)
weeksworkedcols = ['weeksworked00','weeksworked01','weeksworked02',
  'weeksworked03','weeksworked04']

nls97[weeksworkedcols].head(2).T
nls97.shape

# use stack to convert data from wide to long
weeksworked = nls97[weeksworkedcols].\
  stack(dropna=False).\
  reset_index().\
  rename(columns={'level_1':'year',0:'weeksworked'})

weeksworked.head(10)

# Fix the year values
weeksworked['year'] = weeksworked.year.str[-2:].astype(int)+2000
weeksworked.head(10)
weeksworked.shape

# use melt to transform data from wide to long
weeksworked = nls97.reset_index().\
  loc[:,['originalid'] + weeksworkedcols].\
  melt(id_vars=['originalid'], value_vars=weeksworkedcols,
    var_name='year', value_name='weeksworked')

weeksworked['year'] = weeksworked.year.str[-2:].astype(int)+2000
weeksworked.set_index(['originalid'], inplace=True)
weeksworked.loc[[8245,3962]]

# reshape more columns with melt
colenrcols = ['colenroct00','colenroct01','colenroct02',
  'colenroct03','colenroct04']
colenr = nls97.reset_index().\
  loc[:,['originalid'] + colenrcols].\
  melt(id_vars=['originalid'], value_vars=colenrcols,
    var_name='year', value_name='colenr')

colenr['year'] = colenr.year.str[-2:].astype(int)+2000
colenr.set_index(['originalid'], inplace=True)
colenr.loc[[8245,3962]]

# merge the weeks worked and enrollment data
workschool = pd.merge(weeksworked, colenr, on=['originalid','year'], how="inner")
workschool.shape
workschool.set_index(['originalid'], inplace=True)
workschool.loc[[8245,3962]]


'''  9-a4. wide_to_long.py  '''

# import pandas and load the nls data
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97f.csv")
nls97.set_index('personid', inplace=True)

# view some of the weeks worked and college enrollment data
weeksworkedcols = ['weeksworked00','weeksworked01','weeksworked02',
  'weeksworked03','weeksworked04']
colenrcols = ['colenroct00','colenroct01','colenroct02',
  'colenroct03','colenroct04']
nls97.loc[nls97.originalid.isin([1,2]),
  ['originalid'] + weeksworkedcols + colenrcols].T

# run the wide_to_long function
workschool = pd.wide_to_long(nls97[['originalid'] + weeksworkedcols 
  + colenrcols], stubnames=['weeksworked','colenroct'], 
  i=['originalid'], j='year').reset_index()
workschool['year'] = workschool.year+2000
workschool = workschool.sort_values(['originalid','year'])
workschool.set_index(['originalid'], inplace=True)
workschool.head(10)

# run the melt with unaligned suffixes
weeksworkedcols = ['weeksworked00','weeksworked01','weeksworked02',
  'weeksworked04','weeksworked05']
workschool = pd.wide_to_long(nls97[['originalid'] + weeksworkedcols 
  + colenrcols], stubnames=['weeksworked','colenroct'], 
  i=['originalid'], j='year').reset_index()
workschool['year'] = workschool.year+2000
workschool = workschool.sort_values(['originalid','year'])
workschool.set_index(['originalid'], inplace=True)
workschool.head(12)


'''  9-a5. unstack_pivot.py  '''

# import pandas and load the stacked and melted nls data
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
weeksworkedstacked = pd.read_pickle("data/nlsweeksworkedstacked.pkl")
workschoolmelted = pd.read_pickle("data/nlsworkschoolmelted.pkl")

# view the stacked weeks worked data
weeksworkedstacked.head(10)
weeksworkedstacked.index

# use stack to convert from long to wide
weeksworked = weeksworkedstacked.unstack()
weeksworked.head(10)

# use pivot to convert from long to wide
workschoolmelted.loc[workschoolmelted.originalid.isin([1,2])].sort_values(['originalid','year'])
workschool = workschoolmelted.pivot(index='originalid', columns='year', values=['weeksworked','colenroct']).reset_index()
workschool.columns = workschool.columns.map('{0[0]}{0[1]}'.format)
workschool.loc[workschool.originalid.isin([1,2])].T



'''  9-a6. unstack_pivotb.py  '''

# import pandas and load the stacked and melted nls data
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97f.csv")
nls97.set_index(['originalid'], inplace=True)

# stack the data again
weeksworkedcols = ['weeksworked00','weeksworked01',
  'weeksworked02','weeksworked03','weeksworked04']

weeksworkedstacked = nls97[weeksworkedcols].\
  stack(dropna=False)
weeksworkedstacked.loc[[1,2]]

# melt the data again
weeksworkedmelted = nls97.reset_index().\
  loc[:,['originalid'] + weeksworkedcols].\
  melt(id_vars=['originalid'], value_vars=weeksworkedcols,
    var_name='year', value_name='weeksworked')
weeksworkedmelted.loc[weeksworkedmelted.originalid.isin([1,2])].\
  sort_values(['originalid','year'])

# use stack to convert from long to wide
weeksworked = weeksworkedstacked.unstack()
weeksworked.loc[[1,2]]

# use pivot to convert from long to wide
weeksworked = weeksworkedmelted.pivot(index='originalid', \
  columns='year', values=['weeksworked']).reset_index()
weeksworked.columns = ['originalid'] + \
  [col[1] for col in weeksworked.columns[1:]]
weeksworked.loc[weeksworked.originalid.isin([1,2])].T



'''  9-b1. firstlook.py  '''

# import pandas and os and sys libraries
import pandas as pd
import os
import sys
nls97 = pd.read_csv("data/nls97f.csv")
nls97.set_index('personid', inplace=True)

# import the basicdescriptives module
sys.path.append(os.getcwd() + "/helperfunctions")
import basicdescriptives as bd
# import importlib
# importlib.reload(bd)
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 15)

# take a first look at the NLS data
dfinfo = bd.getfirstlook(nls97)
bd.displaydict(dfinfo)

# pass values to the nrows and uniqueid parameters
dfinfo = bd.getfirstlook(nls97,2,'originalid')
bd.displaydict(dfinfo)

# work with some of the dictionary keys and values
dfinfo['nrows']
dfinfo['dtypes']
dfinfo['nrows'] == dfinfo['uniqueids']



'''  9-b2. taking_measure.py  '''

# import the pandas, os, and sys libraries
import pandas as pd
import os
import sys
nls97 = pd.read_csv("data/nls97f.csv")
nls97.set_index('personid', inplace=True)

# import the basicdescriptives module
sys.path.append(os.getcwd() + "/helperfunctions")
import basicdescriptives as bd
# import importlib
# importlib.reload(bd)
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 100)

# show summary statistics for continuous variables
bd.gettots(nls97[['satverbal','satmath']]).T
bd.gettots(nls97.filter(like="weeksworked"))

# count missing per column and per row
missingsbycols, missingsbyrows = bd.getmissings(nls97[['weeksworked16','weeksworked17']], True)
missingsbycols
missingsbyrows
missingsbycols, missingsbyrows = bd.getmissings(nls97[['weeksworked16','weeksworked17']])
missingsbyrows

# do frequencies for categorical columns
nls97.loc[:, nls97.dtypes == 'object'] = \
  nls97.select_dtypes(['object']). \
  apply(lambda x: x.astype('category'))
bd.makefreqs(nls97, "views/nlsfreqs.txt")

# do counts and percentages by groups
bd.getcnts(nls97, ['maritalstatus','gender','colenroct00'])
bd.getcnts(nls97, ['maritalstatus','gender','colenroct00'], "colenroct00.str[0:1]=='1'")






'''  9-b3. check_outliers.py  '''

# import the pandas, os, and sys libraries and load the nls and covid data
import pandas as pd
import os
import sys
import pprint
nls97 = pd.read_csv("data/nls97f.csv")
nls97.set_index('personid', inplace=True)
covidtotals = pd.read_csv("data/covidtotals720.csv")

# import the outliers module
sys.path.append(os.getcwd() + "/helperfunctions")
import outliers as ol
# import importlib
#importlib.reload(ol)
pd.set_option('display.width', 72)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 100)

# get the distribution of a variable
dist = ol.getdistprops(covidtotals.total_cases_pm)
pprint.pprint(dist)

# show outlier rows
sumvars = ['satmath','wageincome']
othervars = ['originalid','highestdegree','gender','maritalstatus']
outliers = ol.getoutliers(nls97, sumvars, othervars)
outliers.varname.value_counts(sort=False)
outliers.loc[outliers.varname=='satmath', othervars + sumvars]
outliers.to_excel("views/nlsoutliers.xlsx")

# do histogram or boxplot of a series
ol.makeplot(nls97.satmath, "Histogram of SAT Math", "SAT Math")
ol.makeplot(nls97.satmath, "Boxplot of SAT Math", "SAT Math", "box")



'''  9-b4. combine_aggregate.py  '''

# import the pandas, os, and sys libraries
import pandas as pd
import os
import sys

# import combineagg module
sys.path.append(os.getcwd() + "/helperfunctions")
import combineagg as ca
# import importlib
# importlib.reload(ca)
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 20)

# load the data frames
coviddaily = pd.read_csv("data/coviddaily720.csv")
ltbrazil = pd.read_csv("data/ltbrazil.csv")
countries = pd.read_csv("data/ltcountries.csv")
locations = pd.read_csv("data/ltlocations.csv")

# summarize panel data by group and time period, with exclusions
ca.adjmeans(coviddaily, 'location','new_cases','casedate')
ca.adjmeans(coviddaily, 'location','new_cases','casedate', 150)

# check matches of merge-by values across data frames
ca.checkmerge(countries.copy(), locations.copy(),\
  "countryid", "countryid")

# concatenate all pickle files in a folder, assuming they have the same structure
landtemps = ca.addfiles("data/ltcountry")
landtemps.country.value_counts()


'''  9-b5. class_cleaning.py  '''

# import the pandas, os, sys, and pprint libraries
import pandas as pd
import os
import sys
import pprint

# import the respondent class
sys.path.append(os.getcwd() + "/helperfunctions")
import respondent as rp
# import importlib
# importlib.reload(rp)
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 100)

# load the NLS data and then create a list of dictionaries
nls97 = pd.read_csv("data/nls97f.csv")
nls97list = nls97.to_dict('records')
nls97.shape
len(nls97list)
pprint.pprint(nls97list[0:1])

# loop through the list creating a respondent instance each time
analysislist = []
for respdict in nls97list:
  resp = rp.Respondent(respdict)
  newdict = dict(originalid=respdict['originalid'],
    childnum=resp.childnum(),
    avgweeksworked=resp.avgweeksworked(),
    age=resp.ageby('20201015'),
    baenrollment=resp.baenrollment())
  analysislist.append(newdict)

# create a pandas data frame
len(analysislist)
resp.respondentcnt
pprint.pprint(analysislist[0:2])
analysis = pd.DataFrame(analysislist)
analysis.head(2)


'''  9-b6. class_cleaning_json.py  '''

# import pandas, json, pprint, and requests
import pandas as pd
import json
import os
import sys
import pprint
import requests
#import importlib
#importlib.reload(ci)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 8)

# import the collection items module
sys.path.append(os.getcwd() + "/helperfunctions")
import collectionitem as ci

# load the art museum's json data
response = requests.get("https://openaccess-api.clevelandart.org/api/artworks/?african_american_artists")
camcollections = json.loads(response.text)
camcollections = camcollections['data']

# loop through the list creating a collection item instance each time
analysislist = []
for colldict in camcollections:
  coll = ci.Collectionitem(colldict)
  newdict = dict(id=colldict['id'],
    title=colldict['title'],
    type=colldict['type'],
    creationdate=colldict['creation_date'],
    ncreators=coll.ncreators(),
    ncitations=coll.ncitations(),
    birthyearsall=coll.birthyearsall(),
    birthyear=coll.birthyearcreator1())
  analysislist.append(newdict)

# create a pandas data frame
len(camcollections)
len(analysislist)
pprint.pprint(analysislist[0:1])
analysis = pd.DataFrame(analysislist)
analysis.birthyearsall.value_counts().head()
analysis.head(2)




'''  merge_imp.py  '''

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 21:15:05 2021

@author: begas05
"""

import glob
import os

path = 'D:/3. 데이터 분석 참고자료/data-cleansing-main'
os.chdir(path)
os.getcwd()


if os.path.exists("merged_source.py"):
    os.remove("merged_source.py")
else:
    print("The file does not exist")

read_files = glob.glob("*.py")

print(read_files)

with open("merged_source.py", "wb") as outfile:
    for f in read_files:
        i = 0
        line = "\n\n" + "'''  " + f + "  '''" + "\n\n"
        i += 1
        outfile.write(line.encode('utf-8'))
        with open(f, "rb") as infile:
            outfile.write(infile.read())
            
            


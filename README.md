# openet
Analysis of modern remote sensing-based ET and comparison with existing DNRC methodology

## File Structure
Highlighting the organization and function of the most important/relevant files for the project.
### In Repository
- **from_swim_rs**: utilities used in ssebop processing
- **iwr**
  - iwr_approx.py: main code that replicates the functionality of IWR software in Python
  - point_comparison.py: very important in development of iwr_approx.py, not used now.
- **reference_et**: files adapted from [here](https://github.com/pyet-org/pyet/tree/master), various meteorological equation helper functions.
  - modified_bcriddle.py: Initial version of modified blaney criddle calculations. Updated versions can be found in iwr_approx.py.
- **run**
  - analysis.py: In progress, will handle all the plotting and analysis of consumptive use results from run_all.py.
  - opnt.py: Handles OpenET data fetching, renaming, and checking. Everything needed before loading data into the database using run_all.py.
  - run_all.py: Handles everything involving building and updating the database. Run after opnt.pt/etof files have prepared.
  - ssebop_processing.py: ssebop, one of the models used by OpenET, can be used as an alternative data source for etof values. This code downloads and processes data through Google Earth Engine, using functions in from_swim_rs.
- **utils**
  - agrimet.py: fetches agrimet data.
  - elevation.py: very simple GEE function to get elevation from coordinates.
  - thredds.py: Handles fetching of gridmet and other data stored online. Please see [this project](https://github.com/MTDNRC-WRD/chmdata/tree/main) for the authoritative version.
- README.md: this document.
### Local File Dependencies
- irrmapper_ref_SID.csv: table of data representing the fraction of a field that was deemed to have been irrigated in any given year, produced using [this code](https://code.earthengine.google.com/562fb670c36c9fdbbef965feeb235780) on GEE. See more information on IrrMapper [here](https://www.mdpi.com/2072-4292/12/14/2328) and [here](https://github.com/dgketchum/EEMapper).
- Statwide Irrigation Dataset: available [here](https://mslservices.mt.gov/Geographic_Information/Data/DataList/datalist_Details.aspx?did=%7Bf33bc611-8d4e-4d92-ae99-49762dec888b%7D) from the Montana State Library.

## How To's
How to get started using this project.

### Python environment setup
...

### Building the database
Hopefully, if you are working with this project, you will simply be expanding an existing database. However, here is how to build the database from scratch.

Start with opnt.py. This process also involves working with several other interfaces to assemble the neceassry data sources.
- Create an [OpenET](https://etdata.org/) account and make an api key, this will be passed to 'openet_get_fields_export' as the content of a txt file.
- The GEE assets used here are available publicly, so these filenames do not need to be changed in order to do the MT statewide analysis. However, if you are new to GEE, you will need to set that up first ([get started here](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api)).
- Run 'get_all_openet_etof_data' to start processing the required files. Wait for these files to show up in your Google Drive, then download them and store them together in a separate directory from other files to prevent errors in reading them into the database later.
- Update the 'path_' variable to point to the directory where you put the etof files. Then run 'rename_etof_downloads' so the filenames are useful.
- Optional: if desired, run 'concat_etof' to create one csv file with each county's data, then run 'check_etof_data_concat' ...

## FYI's
Information on the quirks and particularities of the project.

**On FIPS Codes**: While the general procedure in this project can be used as a guideline for other projects, many components of the code have been very specifically tailored to the needs of the statewide analysis. Most notably, this includes almost every processing component being broken up into chunks by counties, referred to throughout the project by their respective three-digit FIPS code (stored as a string to maintain leading zeros). Several of the larger counties (with respect to number of mapped fields) required splitting in order for processing requests to succeed. These chunks are additionally identified with the letters ‘a’ and ‘b’ appended to the FIPS code. Once the data is ready to be put into the database in run_all.py, these chunks are recombined and these counties do not require special treatment.

**SQLite Structure**: Once preprocessing is complete, and the SQLite database is created and beginning to be populated, the structure of the database is largely set, and would likely require starting over from scratch in order to incorporate additional columns or table constraints. Additional rows within the constraints of the initialized database tables, however, are readily added.

**SQLite Duplicate Data**: The database tables as designed in the code will not update duplicate information or produce an error if data with the same index is generated by the functions. It will be silently ignored, and the previous value will remain, unless the insert functions are specifically altered to override the default behavior of the tables. (Note: 'gridmet_ts' and 'field_data' tables are a possible exception to this.)

## TODO's
Outstanding tasks and ideas for improving the project.

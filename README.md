# MCpredict

Repository contains code for running the Chen magnetic cloud prediction model. See reference Papers: Chen et al. 1996, 1997, 2012
Arge et al. 2002

http://adsabs.harvard.edu/abs/1996GeoRL..23..625C

http://adsabs.harvard.edu/abs/1997JGR...10214701C

http://adsabs.harvard.edu/abs/2002stma.conf..393A

http://adsabs.harvard.edu/abs/2012SpWea..10.4005C


## __Modules__

### MCpredict.py

This module contains functions to read in either real time or historical 
solar wind data and determine geoeffective and non geoeffective "events" 
present in the magnetic field data. An event is defined to be > 120 minutes 
long and with start and end times determined when Bz component of the magentic
field changes sign. 
 
A sinusoid is fitted to the Bz component to predict the event Bz maximum 
(or minimum) value and the event duration, therefore giving some predictive 
diagnostic of whether of not we expect the event to be geoeffective.
 
When applied to historical data, classification of the geoeffectiveness of the events is 
determined by comparison with the Dst value during that event. If Dst < -80 
the event is considered to be geoeffective. If the Dst is > -80 then the event
is considered to be non geoeffective. Events occuring in the wake of geoeffective
events, where the Dst is still recovering are classed as ambigous. 
 
The top level function Chen_MC_Prediction is called to run the model either to
real-time data or to a historical dataset e.g.

    import MCpredict as MC
    import read_dst as dst
    import MC_predict_pdfs as pdf
    import pickle
    
    #read in Dst data for event classification
    dst_data = dst.read_dst_df()
    
    #generate (or eventually read in predetermined) bayesian PDF (see MC_predict_pdfs.py)
    
    events_frac = pickle.load(open("events_frac.p","rb"))
    Pbzm_tau_e_bzmp_taup, norm_bzm_tau_e_bzmp_taup, P0, P1, P1_map = create_pdfs(events_frac, kernel_alg = 'scipy_stats')
    
    data, events, events_frac = MC.Chen_MC_Prediction(start_date, end_date, dst_data, Pbzm_tau_e_bzmp_taup)

The original version of this code is from Jim Chen and Nick Arge
and is called DOACE_hr.pro. This version is the python translation of
IDL code written by Michele Cash of DOACE.pro modifed to 
run in near real-time, to read in data from the SWPC database,
and to make the code more readable by removing goto statements
and 5 minute data averages.

### MC_predict_pdfs.py 

This module comtains functions to generate the PDFs for the Chen geoeffective magnetic cloud prediction Bayesian formulation from historical solar wind data i.e. equation 6 in Chen 1997.
    
P((Bzm, tau) n e|Bzm', tau' ; f) 
= P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
where
    
Bzm = actual value of Bz max for a magnetic cloud
tau = actual duation of a magnetic cloud
Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
tau' = fitted/estimated value of duration at fraction (f) of an event
e = geoeffective event
n = nongeoeffective event
f = fraction of an event
    
The data is stored as a pickle file and should be read in as:
    
    import pickle
    events_frac = pickle.load(open("events_frac.p","rb"))
    
The top level create_pdfs function generates all the input PDFs
and returns the posterior PDF P((Bzm, tau) n e|Bzm', tau' ; f)
along with some other diagnostic values

    import MC_predict_pdfs as pdf
    
    Pbzm_tau_e_bzmp_taup, norm_bzm_tau_e_bzmp_taup, P0, P1, P1_map = create_pdfs(events_frac, kernel_alg = 'scipy_stats')
    

### richardson_mc_analysis.py

This module contains functions to test the cloud fitting routine on the list of known ICMEs with associated magnetic clouds from the Richardson and Cane ICME list. 

Includes some plotting functions to assess the relation between Bzm and tau and how well the fitting routine is working. 

### Richardson_ICME_list.py

Reads in the Richardson and Cane's ICME list from csv file


### Dst_hourly.csv: 
Reads in the hourly Dst data to be used when classifying an event
as geoeffective (Dst < -80 ) of nongeoeffective (Dst > -80)

### read_database.py

Routines for reading in solar wind data from the database or csv file 

### plot_ace.py

Plotting routines for ACE_MAG_1m and ACE_SWEPAM_1M data

## __Data files__

### events.p

Pickle file containing the event data for historical solar wind events

### events_frac.p

Pickle file containing the event data for historical solar wind events

### Dst_hourly.csv

Hourly Dst data

### Richardson_and_Cane_ICME_lists.csv

List of ICME events from Richardson and Cane with indictaion of whether or not a magnetic cloud was associated with the event. Used to test magnetic cloud fitting model



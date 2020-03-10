import utils
import numpy as np
import pandas as pd

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''

    indx_date = pd.merge(events, mortality, how='outer', on='patient_id', suffixes=('_event', '_mortality'))
    indx_date.label.fillna(0, inplace=True, downcast='int64')
    indx_date['timestamp_event'] = pd.to_datetime(indx_date['timestamp_event'])
    indx_date['timestamp_mortality'] = pd.to_datetime(indx_date['timestamp_mortality'])
    indx_date = indx_date[['patient_id','timestamp_event','timestamp_mortality','label']].groupby(['patient_id']).agg('max')
    indx_date['indx_date'] = np.where(indx_date['label'] != 0, indx_date['timestamp_mortality'] - pd.to_timedelta(30, unit='d'), indx_date['timestamp_event'])
    indx_date['indx_date'] = pd.to_datetime(indx_date['indx_date'])
    indx_date = indx_date.reset_index()
    indx_date = indx_date.drop(columns=['timestamp_event', 'timestamp_mortality', 'label'])
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

    merge2 = pd.merge(events, indx_date, how='inner', on='patient_id')
    merge2['timestamp'] = pd.to_datetime(merge2['timestamp'])
    merge3 = merge2[merge2['timestamp'] <= merge2['indx_date']]
    filtered_events = merge3[merge3['timestamp'] >= (merge3['indx_date'] - pd.to_timedelta(2000, unit='d'))]
    filtered_events = filtered_events.drop(columns=['event_description', 'timestamp', 'indx_date'])
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''

    aggregated_events = pd.merge(filtered_events_df, feature_map_df, how='inner', on='event_id')
    aggregated_events = aggregated_events.dropna(subset=['value'])
    aggregated_events = aggregated_events[['patient_id', 'event_id', 'value', 'idx']].groupby(['patient_id','event_id', 'idx']).agg('count').reset_index()
    grouper = aggregated_events.groupby('idx')['value']
    maxes = grouper.transform('max')
    aggregated_events = aggregated_events.assign(value=(aggregated_events.value/maxes))
    aggregated_events = aggregated_events.rename(columns={'idx': 'feature_id', 'value': 'feature_value'})
    aggregated_events = aggregated_events.drop(columns=['event_id'])
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    aggregated_events = aggregated_events.astype({'patient_id': float, 'feature_id': float, 'feature_value': float})
    patient_features = aggregated_events.set_index('patient_id').apply(tuple,1).groupby(level=0).agg(lambda x: list(x.values)).to_dict()
    df = pd.merge(events, mortality, how='outer', on='patient_id', suffixes=('_event', '_mortality'))
    df.label.fillna(0, inplace=True, downcast='int64')
    df = df.astype({'patient_id': float})
    mortality = pd.Series(df.label.values,index=df.patient_id).to_dict()

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')

    for k, v in patient_features.items():
        deliverable1.write(bytes((str(mortality[k])+' '),'UTF-8')); #Use 'UTF-8'
        deliverable2.write(bytes((str(int(k)) + ' ' + str(mortality[k]) + ' '),'UTF-8'));
        v.sort(key=lambda tup: tup[0])
        for p in v:
            deliverable1.write(bytes((str(int(p[0]))+':'+f'{p[1]:.6f}'+' '),'UTF-8'));
            deliverable2.write(bytes((str(int(p[0]))+':'+f'{p[1]:.6f}'+' '),'UTF-8'));
        deliverable1.write(bytes((u'\n'),'UTF-8'));
        deliverable2.write(bytes((u'\n'),'UTF-8'));

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()
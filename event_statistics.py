import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    
    merge = pd.merge(events, mortality, how='outer', on='patient_id', suffixes=('_event', '_mortality'))
    merge.label.fillna(0, inplace=True, downcast='int64')
    dead = merge[merge.label==1]
    alive = merge[merge.label==0]
    dead_counts = dead.groupby(['patient_id']).agg('count')
    alive_counts = alive.groupby(['patient_id']).agg('count')

    avg_dead_event_count = dead_counts.event_id.mean()
    max_dead_event_count = dead_counts.event_id.max()
    min_dead_event_count = dead_counts.event_id.min()
    avg_alive_event_count = alive_counts.event_id.mean()
    max_alive_event_count = alive_counts.event_id.max()
    min_alive_event_count = alive_counts.event_id.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    merge = pd.merge(events, mortality, how='outer', on='patient_id', suffixes=('_event', '_mortality'))
    merge.label.fillna(0, inplace=True, downcast='int64')
    dead = merge[merge.label==1]
    alive = merge[merge.label==0]
    dead_counts = dead.groupby('patient_id').agg('nunique')
    alive_counts = alive.groupby('patient_id').agg('nunique')

    avg_dead_encounter_count = dead_counts.timestamp_event.mean()
    max_dead_encounter_count = dead_counts.timestamp_event.max()
    min_dead_encounter_count = dead_counts.timestamp_event.min() 
    avg_alive_encounter_count = alive_counts.timestamp_event.mean()
    max_alive_encounter_count = alive_counts.timestamp_event.max()
    min_alive_encounter_count = alive_counts.timestamp_event.min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    merge = pd.merge(events, mortality, how='outer', on='patient_id', suffixes=('_event', '_mortality'))
    merge.label.fillna(0, inplace=True, downcast='int64')
    merge['timestamp_event'] = pd.to_datetime(merge['timestamp_event'])
    dead = merge[merge.label==1]
    alive = merge[merge.label==0]
    dead_range = dead[['patient_id','timestamp_event']].groupby(['patient_id']).agg(['max','min'])
    alive_range = alive[['patient_id','timestamp_event']].groupby(['patient_id']).agg(['max','min'])
    dead_range['record_length'] = (dead_range['timestamp_event']['max']-dead_range['timestamp_event']['min']).dt.days
    alive_range['record_length'] = (alive_range['timestamp_event']['max']-alive_range['timestamp_event']['min']).dt.days

    avg_dead_rec_len = dead_range.record_length.mean()
    max_dead_rec_len = dead_range.record_length.max()
    min_dead_rec_len = dead_range.record_length.min()
    avg_alive_rec_len = alive_range.record_length.mean()
    max_alive_rec_len = alive_range.record_length.max()
    min_alive_rec_len = alive_range.record_length.min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()

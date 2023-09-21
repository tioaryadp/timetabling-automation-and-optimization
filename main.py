import numpy as np
import pandas as pd
import math
import random
import streamlit as st

pd.set_option('display.max_rows', 10)
pd.options.mode.chained_assignment = None
scroll_text = None
st.title('Penjadwalan SMPN 1 Jombang')
file_meet = None
file_timeslot = None
file_guru = None
file_pjok = None

if 'status_text' not in st.session_state:
    st.session_state['status_text'] = ''

dataset_meet = None
dataset_timeslot = None
dataset_guru = None
timeslot_PJOK = None

timeslot_assignment = None

hour=0
number_of_hours = 0

crash = []
  
#Check apakah jam telah diinputkan pada timeslot
def not_assigned(rand_id_hour, checklist):
  if rand_id_hour not in checklist:
    return True
  else:
    return False
  
#Generate random number
def generate_random_number(choice_number, checklist):
  rand_variable = random.choice(choice_number)
  while not_assigned(rand_variable, checklist) == False:
    rand_variable = random.choice(choice_number)
  return rand_variable

#Memastikan seluruh id_hour diassignkan pada timetable (HC_1)
def is_all_id_hour_assigned(solution):
  if solution['id_hour'].nunique() == number_of_hours:
    return True
  else:
    return False
  
#Memastikan bahwa pelajaran PJOK diassign pada jam pagi (HC_2)
def is_pjok_in_morning(solution):
  if all(item == 1 for item in solution['id_time'].loc[solution['MP'] == 9].values.tolist()):
    return True
  else:
    return False
  
#Update data in row
def update_row(timetable, row_numb):
  timetable.at[row_numb, 'id_meet'] = dataset_meet['id_meet'].loc[dataset_meet['id_hour'] == timetable.at[row_numb, 'id_hour']].values
  timetable.at[row_numb, 'MP'] = dataset_meet['MP'].loc[dataset_meet['id_hour'] == timetable.at[row_numb, 'id_hour']].values
  timetable.at[row_numb, 'pertemuan'] = dataset_meet['pertemuan'].loc[dataset_meet['id_hour'] == timetable.at[row_numb, 'id_hour']].values
  timetable.at[row_numb, 'id_kelas_y'] = dataset_meet['id_kelas'].loc[dataset_meet['id_hour'] == timetable.at[row_numb, 'id_hour']].values

def delete_data(solution):
  timeslot_assignment['id_hour'].loc[timeslot_assignment['id_kelas'] == k] = 0
  timeslot_assignment['id_meet'].loc[timeslot_assignment['id_kelas'] == k] = 0
  timeslot_assignment['MP'].loc[timeslot_assignment['id_kelas'] == k] = 0
  timeslot_assignment['pertemuan'].loc[timeslot_assignment['id_kelas'] == k] = 0
  timeslot_assignment['id_kelas_y'].loc[timeslot_assignment['id_kelas'] == k] = 0
  
#Memastikan tidak ada guru yang mendapatkan jadwal crash selain guru PJOK dapat crash max. 2 kelas (HC_3)
def is_no_crash(solution):
  guru_check = []
  guru_PJOK = list(set(solution['id_guru'].loc[solution['MP'] == 9].values.tolist()))
  for g in solution['id_guru'].unique():
    if g in guru_PJOK:
      jp_list = solution['id_jp'].loc[solution['id_guru'] == g].values.tolist()
      uniqlist = []
      duplist = []
      for i in jp_list:
        if i not in uniqlist:
          uniqlist.append(i)
        else:
          duplist.append(i)

      dup_tolerence = []
      for x in duplist:
        if jp_list.count(x) > 2:
          dup_tolerence.append(0)
        else:
          dup_tolerence.append(1)
      if all(item == 1 for item in dup_tolerence):
        guru_check.append(1)
      else:
        guru_check.append(0)

    elif g not in guru_PJOK:
      if(len(set(solution['id_jp'].loc[solution['id_guru'] == g].values.tolist())) == len(solution['id_jp'].loc[solution['id_guru'] == g].values.tolist())):
        guru_check.append(1)
      else:
        guru_check.append(0)

  if all(item == 1 for item in guru_check):
    return True
  else:
    return False
  
#Memastikan tidak ada meet yang terpisah hari (HC_4)
def is_all_meet_not_separated(solution):
  for m in solution['id_meet'].unique():
    if all(item == solution['id_day'].loc[solution['id_meet'] == m].values.tolist()[0] for item in solution['id_day'].loc[solution['id_meet'] == m].values.tolist()) == False:
      return False
  return True

#Memastikan setiap harinya tidak ada id_meet dengan MP yang sama (HC_5)
def is_no_same_MP(solution):
  no_same_MP_check = []
  for d in solution['id_day'].unique():
    for k in solution['id_kelas'].unique():
      timeslot_d_k = solution[['id_meet','MP']].loc[(solution['id_day'] == d) & (solution['id_kelas'] == k)]
      MP_check = []
      for p in timeslot_d_k['MP'].unique():
        if all(item == timeslot_d_k['id_meet'].loc[timeslot_d_k['MP'] == p].values.tolist()[0] for item in timeslot_d_k['id_meet'].loc[timeslot_d_k['MP'] == p].values.tolist()):
          MP_check.append(1)
        else:
          MP_check.append(0)
      if all(item == 1 for item in MP_check):
        no_same_MP_check.append(1)
      else:
        no_same_MP_check.append(0)

  if all(item == 1 for item in no_same_MP_check):
    return True
  else:
    return False
  
#Cek apakah solusi yang dihasilkan feasible
def is_feasible(solution):
  if is_all_id_hour_assigned(solution) == True:
    if is_pjok_in_morning(solution) == True:
      if is_no_crash(solution) == True:
        if is_all_meet_not_separated(solution) == True:
          if is_no_same_MP(solution) == True:
            return True
          else:
            return False
        else:
          return False
      else:
        return False
    else:
      return False
  else:
    return False
  
#Memastikan tidak ada guru yang mendapatkan jadwal crash selain guru PJOK dapat crash max. 2 kelas (HC_3 per class)
def class_is_no_crash(solution):
  guru_check = []
  guru_PJOK = list(set(solution['id_guru'].loc[solution['MP'] == 9].values.tolist()))
  g_list = solution['id_guru'].unique().tolist()
  g_list = [gl for gl in g_list if gl != 0]
  for g in g_list:
    if g in guru_PJOK:
      jp_list = solution['id_jp'].loc[solution['id_guru'] == g].values.tolist()
      uniqlist = []
      duplist = []
      for t in jp_list:
        if t not in uniqlist:
          uniqlist.append(i)
        else:
          duplist.append(i)

      dup_tolerence = []
      for x in duplist:
        if jp_list.count(x) > 2:
          dup_tolerence.append(0)
        else:
          dup_tolerence.append(1)
      if all(item == 1 for item in dup_tolerence):
        guru_check.append(1)
      else:
        guru_check.append(0)

    elif g not in guru_PJOK:
      if(len(set(solution['id_jp'].loc[solution['id_guru'] == g].values.tolist())) == len(solution['id_jp'].loc[solution['id_guru'] == g].values.tolist())):
        guru_check.append(1)
      else:
        guru_check.append(0)

  if all(item == 1 for item in guru_check):
    return True
  else:
    return False
  
def crash_report(solution):  
  guru_crash = []
  guru_PJOK = list(set(solution['id_guru'].loc[solution['MP'] == 9].values.tolist()))
  for g in solution['id_guru'].unique():
    if g in guru_PJOK:
      jp_list = solution['id_jp'].loc[solution['id_guru'] == g].values.tolist()
      uniqlist = []
      duplist = []
      for i in jp_list:
        if i not in uniqlist:
          uniqlist.append(i)
        else:
          duplist.append(i)

      dup_tolerence = []
      for x in duplist:
        if jp_list.count(x) > 2:
          dup_tolerence.append(0)
        else:
          dup_tolerence.append(1)
      if all(item == 1 for item in dup_tolerence) == False:
        guru_crash.append(g)

    elif g not in guru_PJOK:
      if(len(set(solution['id_jp'].loc[solution['id_guru'] == g].values.tolist())) == len(solution['id_jp'].loc[solution['id_guru'] == g].values.tolist())) == False:
        guru_crash.append(g)
  return guru_crash

def generate_guru(choice_number, checklist):
  rand_variable = random.choice(choice_number)
  attemp = 0
  while not_assigned(rand_variable, checklist) == False:
    rand_variable = random.choice(choice_number)
    attemp +=1
    if attemp > 50:
      rand_variable = len(data_guru['id_guru'])+1
      data_guru.loc[len(data_guru.index)] = [rand_variable, init_solution.at[i-1, 'MP']]
      attemp = 0
  return rand_variable

def delete_data_guru(solution):
  for cr in crash:
    solution['id_guru'].loc[(solution['id_guru'] == cr) & (solution['id_kelas'] == k)] = 0

#Penalti untuk MIPA yang diassign selain jam pagi (SC_1)
def sc_1(solution):
  SC_penalty = 0
  MIPA = [4,5,6]
  for i in solution['id_hour']:
    if solution.at[i-1, 'MP'] in MIPA:
      if solution.at[i-1, 'id_time'] != 1:
        SC_penalty +=1
  return SC_penalty

#Penalti untuk jeda/jarak pada jadwal mengajar guru (SC_2)
def sc_2(solution):
  SC_penalty = 0
  for g in solution['id_guru'].unique():
    slot_guru = []
    for h in solution['id_hour']:
      if solution.at[h-1, 'id_guru'] == g:
        slot_guru.append(h)
    slot_guru = sorted(slot_guru)
    for i in range(len(slot_guru)-1):
      for j in range(len(slot_guru)-1):
        j+=i+1
        if(j > len(slot_guru)-1):
          break
        if (1 < (slot_guru[j] - slot_guru[i]) <= 5):
          SC_penalty += 1
  return SC_penalty

#Penalti untuk MP yang dijadwalkan dihari yang berurutan (SC_3)
def sc_3(solution):
  SC_penalty = 0
  for mp in solution['MP'].unique():
    slot_MP = []
    for k in solution['id_kelas'].unique():
      for d in solution['id_day'].unique():
        timeslot_d_k = solution[['id_kelas','id_day','MP']].loc[(solution['id_day'] == d) & (solution['id_kelas'] == k)]
        if mp in timeslot_d_k['MP'].values.tolist():
          slot_MP.append(d)
    for i in range(len(slot_MP)-1):
        for j in range(len(slot_MP)-1):
          j+=i+1
          if(j > len(slot_MP)-1):
            break
          if ((slot_MP[j] - slot_MP[i]) == 1):
            SC_penalty += 1
  return SC_penalty

#Penalti untuk setiap penambahan guru
def sc_4(solution):
  SC_penalty = ((len(solution['id_guru'].unique())) - number_of_guru)
  return SC_penalty

#Fungsi Tujuan
def obj_function(solution):
  SC_score = (sc_1(solution)*0.2) + (sc_2(solution)*0.3) + (sc_3(solution)*0.1) + (sc_4(solution)*0.4)
  return SC_score/(number_of_class*6)

def generate_new_solution(solution):
  loop = False
  while not loop:
    proc_sol = solution.copy()
    #Get id_meet for Swap Hour
    swap_random = random.randint(1, len(proc_sol['id_meet'].unique()))
    swap_hour = dataset_meet['id_hour'].loc[dataset_meet['id_meet'] == swap_random].values.tolist()

    #Get Replaced id_meet for Swap
    class_filtered = dataset_meet.loc[dataset_meet['id_kelas'] == int(dataset_meet['id_kelas'].loc[dataset_meet['id_meet'] == swap_random].unique())]
    chosen_candidate = class_filtered['id_meet'].value_counts().loc[lambda s: s == len(swap_hour)].index.tolist()
    meet_chosen = random.choice(chosen_candidate)
    chosen_hour = dataset_meet['id_hour'].loc[dataset_meet['id_meet'] == meet_chosen].values.tolist()

    #Swapping id_meet
    proc_sol['id_hour'].loc[proc_sol['id_meet'] == swap_random] = 0
    for i, row in proc_sol.loc[proc_sol['id_meet'] == meet_chosen].iterrows():
      proc_sol.at[i, 'id_hour'] = generate_random_number(swap_hour, proc_sol['id_hour'].values.tolist())
      update_row(proc_sol, i)
    for j, row in proc_sol.loc[proc_sol['id_hour'] == 0].iterrows():
      proc_sol.at[j, 'id_hour'] = generate_random_number(chosen_hour, proc_sol['id_hour'].values.tolist())
      update_row(proc_sol, j)
    loop = is_feasible(proc_sol)
  return proc_sol

# Simmulated Annealing
def generate_new_solution_sa(solution, itter):
    #Set Parameter SA
    initial_temperature = 1000
    cooling = 0.7
    no_improvement = 0
    no_improve_thres = 10

    # Set Initial Variable SA
    initial_solution = solution
    current_solution = solution
    best_solution = solution

    abs_best_solution = best_solution
    best_fitness = obj_function(best_solution)
    abs_best_fitness = best_fitness

    record_best_fitness = []
    record_temperature = []

    current_temperature = initial_temperature
    eksponential = 1.0

    #Simulated Annealing Algorithm
    for i in range(itter):

        new_timeslot_assignment = generate_new_solution(best_solution)
        current_solution = new_timeslot_assignment
        current_fitness = obj_function(current_solution)
        record_temperature.append(current_temperature)

        if current_fitness < best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness

            if best_fitness < abs_best_fitness:
                abs_best_solution = best_solution
                abs_best_fitness = best_fitness

            no_improvement = 0
            current_temperature = current_temperature*cooling

        elif current_fitness == best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness
    
            no_improvement+=1
            current_temperature = current_temperature*cooling

        else:
            dif_fitness = -abs(current_fitness - best_fitness)
            random_number_2 = random.uniform(0,1)
            eksponential = math.exp(dif_fitness/current_temperature)
            if(random_number_2 <= eksponential):
                best_solution = current_solution
                best_fitness = current_fitness

                if best_fitness < abs_best_fitness:
                    abs_best_solution = best_solution
                    abs_best_fitness = best_fitness

                no_improvement = 0
                current_temperature = current_temperature*cooling

            else:
                no_improvement+=1
                current_temperature = current_temperature*cooling
  
        if no_improvement == no_improve_thres:
            current_temperature = initial_temperature

        st.session_state['status_text'] = ('\niteration: {}, current_temperature: {}, chance: {}, current_fitness: {}'.format(i, current_temperature, eksponential, best_fitness))
        with scroll_text.container():
          st.write(st.session_state['status_text'])
        record_best_fitness.append(best_fitness)

    sum_penalty_init = (sc_1(init_solution)) + (sc_2(init_solution)) + (sc_3(init_solution)) + (sc_4(init_solution))

    sum_penalty = (sc_1(abs_best_solution)) + (sc_2(abs_best_solution)) + (sc_3(abs_best_solution)) + (sc_4(abs_best_solution))

    print('best_fitness: {}, best_solution: {}'.format(abs_best_fitness, abs_best_solution['id_hour'].values.tolist()))

    return abs_best_solution

def generate_new_solution_gd(solution, itter):
    global scroll_text
   #Setting Great Deluge
    raindspeed = 1.2

    #Great Deluge Algorithm
    initial_solution_GD = solution
    current_solution_GD = solution
    current_fitness_GD = obj_function(current_solution_GD)
    best_solution_GD = solution
    best_fitness_GD = current_fitness_GD
    abs_best_fitness_GD = best_fitness_GD
    abs_best_solution_GD = best_solution_GD
    record_best_fitness_GD = []
    record_water_level = []

    water_level = current_fitness_GD
    decay_rate = current_fitness_GD*(raindspeed/itter)

    for i in range(itter):
        current_solution_GD = generate_new_solution(best_solution_GD)
        current_fitness_GD = obj_function(current_solution_GD)
        record_water_level.append(water_level)

        if current_fitness_GD <= best_fitness_GD:
            best_solution_GD = current_solution_GD
            best_fitness_GD = current_fitness_GD
            if abs_best_fitness_GD > best_fitness_GD:
                abs_best_solution_GD = best_solution_GD
                abs_best_fitness_GD = best_fitness_GD

        elif current_fitness_GD <= water_level:
            best_solution_GD = current_solution_GD
            best_fitness_GD = current_fitness_GD
            if abs_best_fitness_GD > best_fitness_GD:
                abs_best_solution_GD = best_solution_GD
                abs_best_fitness_GD = best_fitness_GD

        else:
            current_solution_GD = best_solution_GD
            current_fitness_GD = best_fitness_GD

        water_level = water_level-decay_rate

        st.session_state['status_text'] = '\n\titeration: {}, water_level: {}, current_fitness: {}'.format(i, water_level, best_fitness_GD)
        with scroll_text.container():
          st.write(st.session_state['status_text'])
        record_best_fitness_GD.append(best_fitness_GD)
    
    return abs_best_solution_GD

with st.form("submit_form"):
    file_meet = st.file_uploader("Upload Dataset Meet", type=["xlsx"])
    file_timeslot = st.file_uploader("Upload Dataset Timeslot", type=["xlsx"])
    file_guru = st.file_uploader("Upload Dataset Guru", type=["xlsx"])
    file_pjok = st.file_uploader("Upload Dataset PJOK", type=["xlsx"])
    number_of_itter = st.number_input("Number of Itteration", min_value=1, step=1)
    algorithm = st.selectbox(
    'Select Algorithm',
    ('Simmulated Annealing', 'Great Deluge'))

    submitted = st.form_submit_button("Submit")
    if submitted:
        if file_meet is None or file_timeslot is None or file_guru is None or file_pjok is None :
            st.write('Please upload all the data')
        
        else:
            st.write('Run Progress : ')
            scroll_text = st.empty()
            st.session_state['status_text'] = ''
            with scroll_text.container():
                st.write(st.session_state['status_text']) 

            dataset_meet = pd.read_excel(file_meet)
            dataset_timeslot = pd.read_excel(file_timeslot)
            dataset_guru = pd.read_excel(file_guru, usecols='A,B')
            timeslot_PJOK = pd.read_excel(file_pjok)
            data_guru = dataset_guru.copy()

            number_of_timeslot = dataset_timeslot.shape[0]
            number_of_hours = dataset_meet.shape[0]
            number_of_class = dataset_meet['id_kelas'].nunique()
            number_of_guru = dataset_guru.shape[0]

            hour = [0 for i in range(number_of_hours)]
            timeslot_assignment = dataset_timeslot.assign(id_hour = hour)

            timeslot_assignment = dataset_timeslot.assign(id_hour = hour)
            #Assign id_hour ke timeslot
            triple_id_meet = dataset_meet['id_meet'].value_counts().loc[lambda s: s == 3].index.tolist()
            double_id_meet = dataset_meet['id_meet'].value_counts().loc[lambda s: s == 2].index.tolist()
            single_id_meet = dataset_meet['id_meet'].value_counts().loc[lambda s: s == 1].index.tolist()

            #Memastikan bahwa id_kelas == id_kelas_y
            for k in timeslot_assignment['id_kelas'].unique():
                all_id_hour = dataset_meet['id_hour'].loc[dataset_meet['id_kelas'] == k].values.tolist()
                triple_id_hour = dataset_meet['id_hour'].loc[dataset_meet['id_kelas'] == k][dataset_meet['id_meet'].isin(triple_id_meet)].values.tolist()
                double_id_hour = dataset_meet['id_hour'].loc[dataset_meet['id_kelas'] == k][dataset_meet['id_meet'].isin(double_id_meet)].values.tolist()
                single_id_hour = dataset_meet['id_hour'].loc[dataset_meet['id_kelas'] == k][dataset_meet['id_meet'].isin(single_id_meet)].values.tolist()
                loop = False
                attemp = 0
                while not loop:
                    for t in timeslot_PJOK['id_timeslot']:
                        PJOK_slot = timeslot_PJOK['id_hour'].loc[timeslot_PJOK['id_timeslot'] == t].values
                        timeslot_assignment['id_hour'].loc[timeslot_assignment['id_timeslot'] == t] = PJOK_slot
                        update_row(timeslot_assignment, t-1)
                
                    for d in timeslot_assignment['id_day'].unique():
                        for i in timeslot_assignment['id_timeslot'].loc[(timeslot_assignment['id_kelas'] == k) & (timeslot_assignment['id_day'] == d)]:
                            if i == 1 and timeslot_assignment.at[i-1, 'id_hour'] == 0:
                                timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(dataset_meet['id_hour'].loc[dataset_meet['id_kelas'] == k].values.tolist(), timeslot_assignment['id_hour'].values.tolist())
                                update_row(timeslot_assignment, i-1)
                    
                            elif i > 1 and timeslot_assignment.at[i-1, 'id_hour'] == 0:
                                meet = timeslot_assignment.at[i-2, 'id_meet']
                                same_meet = dataset_meet['id_hour'].loc[dataset_meet['id_meet'] == meet].values.tolist()

                                #Memastikan bahwa id_hour yang memiliki id_meet yang sama akan di assign berurutan
                                if all(item in timeslot_assignment['id_hour'].values.tolist() for item in same_meet) == False:
                                    timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(same_meet, timeslot_assignment['id_hour'].values.tolist())
                                    update_row(timeslot_assignment, i-1)
            
                                #Memastikan bahwa tidak ada meet yang terpisah hari
                                elif all(item in timeslot_assignment['id_hour'].values.tolist() for item in same_meet) == True:
                                    same_meet = []
                                    if timeslot_assignment['id_hour'].loc[(timeslot_assignment['id_kelas'] == k) & (timeslot_assignment['id_day'] == d)].values.tolist().count(0) > 3:
                                        timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(triple_id_hour + double_id_hour, timeslot_assignment['id_hour'].values.tolist())
                                        update_row(timeslot_assignment, i-1)
                                    elif timeslot_assignment['id_hour'].loc[(timeslot_assignment['id_kelas'] == k) & (timeslot_assignment['id_day'] == d)].values.tolist().count(0) == 3:
                                        if all(item in timeslot_assignment['id_hour'].values.tolist() for item in triple_id_hour) == False:  
                                            timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(triple_id_hour, timeslot_assignment['id_hour'].values.tolist())
                                            update_row(timeslot_assignment, i-1)
                                        elif all(item in timeslot_assignment['id_hour'].values.tolist() for item in triple_id_hour) == True:
                                            timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(double_id_hour + single_id_hour, timeslot_assignment['id_hour'].values.tolist())
                                            update_row(timeslot_assignment, i-1)

                                    elif timeslot_assignment['id_hour'].loc[(timeslot_assignment['id_kelas'] == k) & (timeslot_assignment['id_day'] == d)].values.tolist().count(0) == 2:
                                        if all(item in timeslot_assignment['id_hour'].values.tolist() for item in double_id_hour) == False:
                                            timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(double_id_hour, timeslot_assignment['id_hour'].values.tolist())
                                            update_row(timeslot_assignment, i-1)
                                        elif all(item in timeslot_assignment['id_hour'].values.tolist() for item in double_id_hour) == True:
                                            timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(single_id_hour, timeslot_assignment['id_hour'].values.tolist())
                                            update_row(timeslot_assignment, i-1)
                        
                                    elif timeslot_assignment['id_hour'].loc[(timeslot_assignment['id_kelas'] == k) & (timeslot_assignment['id_day'] == d)].values.tolist().count(0) == 1:
                                        timeslot_assignment.at[i-1, 'id_hour'] = generate_random_number(single_id_hour, timeslot_assignment['id_hour'].values.tolist())
                                        update_row(timeslot_assignment, i-1)

                    #Cek hingga HC_3 dan HC_5 terpenuhi
                    loop = is_no_same_MP(timeslot_assignment)
                    if loop == False:
                        delete_data(timeslot_assignment)
                        attemp+=1
                        st.session_state['status_text'] =("\n\t\tSearching with attempt: %s"%(attemp))
                        with scroll_text.container():
                          st.write(st.session_state['status_text'])

            init_solution = timeslot_assignment.copy()

            guru = [0 for i in range(number_of_hours)]
            init_solution = init_solution.assign(id_guru = guru)

            for k in init_solution['id_kelas'].unique():
                loop = False
                attemp = 0
                while not loop:
                    gmp_kelas = pd.DataFrame({'MP': dataset_meet['MP'].unique(), 'id_guru': [0 for o in range(len(dataset_meet['MP'].unique()))]})
                    for i in init_solution['id_timeslot'].loc[(init_solution['id_kelas'] == k)]:
                        if i == 1 and init_solution.at[i-1, 'id_guru'] == 0:
                            meet_now = init_solution.at[i-1, 'id_meet']
                            init_solution.at[i-1, 'id_guru'] = random.choice(data_guru['id_guru'].loc[data_guru['MP'] == init_solution.at[i-1, 'MP']].values.tolist())
                            gmp_kelas['id_guru'].loc[gmp_kelas['MP'] == init_solution.at[i-1, 'MP']] = init_solution.at[i-1, 'id_guru']
                        elif i > 1 and init_solution.at[i-1, 'id_guru'] == 0:
                            if init_solution.at[i-1, 'id_meet'] == meet_now:
                                init_solution.at[i-1, 'id_guru'] = init_solution.at[i-2, 'id_guru']
                            else:
                                meet_now = init_solution.at[i-1, 'id_meet']
                                if (gmp_kelas['id_guru'].loc[gmp_kelas['MP'] == init_solution.at[i-1, 'MP']] == 0).bool() == True:
                                    guru_checklist = []
                                    meet_len = len(dataset_meet.loc[dataset_meet['id_meet'] == init_solution.at[i-1, 'id_meet']].index)
                                    for jp in range(meet_len):
                                        guru_jp = init_solution['id_guru'].loc[init_solution['id_jp'] == init_solution.at[i+jp-2, 'id_jp']].values.tolist()
                                        for x in range(len(guru_jp)):
                                            guru_checklist.append(guru_jp[x])
                                    init_solution.at[i-1, 'id_guru'] = generate_guru(data_guru['id_guru'].loc[data_guru['MP'] == init_solution.at[i-1, 'MP']].values.tolist(), np.unique(guru_checklist))
                                    gmp_kelas['id_guru'].loc[gmp_kelas['MP'] == init_solution.at[i-1, 'MP']] = init_solution.at[i-1, 'id_guru']
                                elif (gmp_kelas['id_guru'].loc[gmp_kelas['MP'] == init_solution.at[i-1, 'MP']] == 0).bool() == False:
                                    init_solution.at[i-1, 'id_guru'] = gmp_kelas['id_guru'].loc[gmp_kelas['MP'] == init_solution.at[i-1, 'MP']]
        
                    loop = class_is_no_crash(init_solution)
                    if loop == False:
                        crash = crash_report(init_solution)
                        crash = [c for c in crash if c != 0]
                        delete_data_guru(init_solution)
                        attemp+=1
                        if attemp > 50:
                            get_crash = random.choice(crash)
                            data_guru.loc[len(data_guru.index)] = [len(data_guru['id_guru'])+1, int(data_guru['MP'].loc[data_guru['id_guru'] == get_crash])]
                            attemp = 0
                        st.session_state['status_text'] = ("\n\t\tSearching with attempt: %s %s")%(attemp,crash)
                        with scroll_text.container():
                          st.write(st.session_state['status_text'])
            is_feasible(init_solution)

            df_res = None
            if algorithm == "Simmulated Annealing":
                df_res = generate_new_solution_sa(init_solution, number_of_itter)
            elif algorithm == "Great Deluge":
                df_res = generate_new_solution_gd(init_solution, number_of_itter)
            
            if df_res is not None:

                hari1 = ['Senin', 'Selasa', 'Rabu', 'Kamis']
                hari_jam1 = [f"{h} {i}" for h in hari1 for i in range(1, 10)]

                hari2 = ['Jumat']
                hari_jam2 = [f"{h} {i}" for h in hari2 for i in range(1, 6)]

                hari3 = ['Sabtu']
                hari_jam3 = [f"{h} {i}" for h in hari3 for i in range(1, 8)]

                hari_jam = hari_jam1 + hari_jam2 + hari_jam3

                meet_list_7 = df_res.loc[:479, 'id_hour'].values.tolist()
                meet_list_8 = df_res.loc[480:959, 'id_hour'].values.tolist()
                meet_list_9 = df_res.loc[960:, 'id_hour'].values.tolist()

                nilai_map = {
                    1: 'AGM',
                    2: 'PKN',
                    3: 'BIND',
                    4: 'MAT',
                    5: 'IPA',
                    6: 'IPS',
                    7: 'BING',
                    8: 'MUL-K',
                    9: 'PJOK',
                    10: 'TIK',
                    11: 'SBK',
                    12: 'BJW',
                    13: 'MUL-D',
                    14: 'BK'
                }
                df_res['MP'] = df_res['MP'].replace(nilai_map)
                df_res['pertemuan'] = df_res['pertemuan'].astype(int)
                # Kelas 7
                st.text("Kelas 7")
                final_assignment7 = []
                for i in range(len(meet_list_7)):
                    id = meet_list_7[i]
                    
                    guru_value = df_res.loc[df_res['id_hour'] == id][['id_guru']].values[0][0]
                    mp_value = df_res.loc[df_res['id_hour'] == id][['MP']].values[0][0]
                    pertemuan = df_res.loc[df_res['id_hour'] == id][['pertemuan']].values[0][0]
                    final_assignment7.append (f"{mp_value}|{guru_value}|{pertemuan}")
                    

                # Mengkonversi list menjadi numpy array dengan bentuk 10 baris dan 48 kolom
                my_array = np.array(final_assignment7).reshape(10, 48)

                # Mengkonversi numpy array menjadi dataframe + transpose
                final_assignment7_df = pd.DataFrame(my_array)
                final_assignment7_df = final_assignment7_df.T    

                # #ngasih nama kolom & baris
                final_assignment7_df.columns = ['7A', '7B', '7C', '7D', '7E', '7F', '7G', '7H', '7I', '7J']
                final_assignment7_df.index = hari_jam

                st.dataframe(final_assignment7_df)

                # Kelas 8
                final_assignment8 = []
                for i in range(len(meet_list_8)):
                    id = meet_list_8[i]
                    
                    guru_value = df_res.loc[df_res['id_hour'] == id][['id_guru']].values[0][0]
                    mp_value = df_res.loc[df_res['id_hour'] == id][['MP']].values[0][0]
                    pertemuan = df_res.loc[df_res['id_hour'] == id][['pertemuan']].values[0][0]

                    final_assignment8.append (f"{mp_value}|{guru_value}|{pertemuan}")
                    

                # Mengkonversi list menjadi numpy array dengan bentuk 10 baris dan 48 kolom
                my_array = np.array(final_assignment8).reshape(10, 48)

                # Mengkonversi numpy array menjadi dataframe + transpose
                final_assignment8_df = pd.DataFrame(my_array)
                final_assignment8_df = final_assignment8_df.T    

                # #ngasih nama kolom & baris
                final_assignment8_df.columns = ['8A', '8B', '8C', '8D', '8E', '8F', '8G', '8H', '8I', '8J']
                final_assignment8_df.index = hari_jam

                st.dataframe(final_assignment8_df)

                # Kelas 9
                st.text("Kelas 9")
                final_assignment9 = []
                for i in range(len(meet_list_9)):
                    id = meet_list_9[i]
                    
                    guru_value = df_res.loc[df_res['id_hour'] == id][['id_guru']].values[0][0]
                    mp_value = df_res.loc[df_res['id_hour'] == id][['MP']].values[0][0]
                    pertemuan = df_res.loc[df_res['id_hour'] == id][['pertemuan']].values[0][0]

                    final_assignment9.append (f"{mp_value}|{guru_value}|{pertemuan}")
                    

                # Mengkonversi list menjadi numpy array dengan bentuk 10 baris dan 48 kolom
                my_array = np.array(final_assignment9).reshape(10, 48)

                # Mengkonversi numpy array menjadi dataframe + transpose
                final_assignment9_df = pd.DataFrame(my_array)
                final_assignment9_df = final_assignment9_df.T    

                # #ngasih nama kolom & baris
                final_assignment9_df.columns = ['9A', '9B', '9C', '9D', '9E', '9F', '9G', '9H', '9I', '9J']
                final_assignment9_df.index = hari_jam

                st.dataframe(final_assignment9_df)
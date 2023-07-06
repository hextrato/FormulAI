import random 
import pandas as pd

class FAIGenerator():

    def __init__(self):
        if False:
            raise Exception('[HXT:FAI:Err-000] Never raise this exception.')
        self._RANDOM_LABEL_SEQUENCE = ['P','Q','R','S','T','V','P','Q','R','S','T','P','Q','R','S','P','Q','R','P','Q','P','V']
        self._NO_CATEG_ATTS = 8
        self._NO_CONTI_ATTS = 8
        self._MAX_ATTS_VALUES = 3
        self._MIN_SAMPLE_SIZE = 8
        self._MAX_SAMPLE_SIZE = 8
        self._RANDOM_NOISE = 0.0
        self._VERBOSE = False
        self._TEST_SAMPLE_SIZE = 2
        self._current_label_position = -1
        self._current_sample_size = -1
        self._template = None
        self._dataset = None
    
    def set_random_noise(self,noise_ratio):
        if noise_ratio < 0.0 or noise_ratio > 1.0:
            raise Exception('[HXT:FAI:Err-001] Random noise ration MUST be between 0 and 1.')
        self._RANDOM_NOISE = noise_ratio

    def set_sample_size(self,min_sample,max_sample):
        self._MIN_SAMPLE_SIZE = min_sample
        self._MAX_SAMPLE_SIZE = max_sample

    def set_test_sample_size(self,test_sample_size):
        self._TEST_SAMPLE_SIZE = test_sample_size
        
    def verbose(self):
        self._VERBOSE = True

    def reset_template(self):
        self._template = ['R' for i in range(self._NO_CATEG_ATTS+self._NO_CONTI_ATTS)] 

    def get_template(self):
        return self._template

    def reset_dataset(self):
        self._dataset = {}
        self._dataset["train"] = {'sample': [], 'label': []}
        for f_idx in range(len(self._ATTS_VALUES)):
            self._dataset["train"]['F'+str(f_idx)] = []
        self._dataset["valid"] = {'sample': [], 'label': []}
        for f_idx in range(len(self._ATTS_VALUES)):
            self._dataset["valid"]['F'+str(f_idx)] = []
        self._dataset["test"] = {'sample': [], 'label': []}
        for f_idx in range(len(self._ATTS_VALUES)):
            self._dataset["test"]['F'+str(f_idx)] = []
    
    def set_attributes(self,categorical,continuous):
        if categorical < 1 or continuous < 1:
            raise Exception('[HXT:FAI:Err-002] Number of continuous and categorical attributes MUST be >= 1 each.')
        if categorical > 100 or continuous > 100:
            raise Exception('[HXT:FAI:Err-002] Number of continuous and categorical attributes MUST be <= 100 each.')
        self._NO_CATEG_ATTS = categorical
        self._NO_CONTI_ATTS = continuous
        self._ATTS_VALUES = [] # [1,1,2,2,3,3,4,4,1,1,2,2,3,3,4,4]
        # max_values = 1
        # self._MAX_ATTS_VALUES
        while len(self._ATTS_VALUES) < categorical:
            self._ATTS_VALUES.append(self._MAX_ATTS_VALUES)
        while len(self._ATTS_VALUES) < categorical+continuous:
            self._ATTS_VALUES.append(self._MAX_ATTS_VALUES)
        #print(self._NO_CATEG_ATTS,self._NO_CONTI_ATTS,self._MAX_ATTS_VALUES)
        
    def set_template_value(self,index,value):
        self._template[index] = value
            
    def fill_template_random_values(self):
        for t_idx in range(len(self._template)):
            if str(self._template[t_idx]) == 'R':
                new_random_value = 0
                if random.random() < self._RANDOM_NOISE:
                    new_random_value = random.randrange(self._ATTS_VALUES[t_idx]+1,10)
                self.set_template_value(t_idx,new_random_value)
        for t_idx in range(len(self._template)):
            if t_idx < self._NO_CATEG_ATTS: 
                self._template[t_idx] = 'V'+str(self._template[t_idx])
            else:
                if random.random() < self._RANDOM_NOISE:
                    self._template[t_idx] = round(self._template[t_idx] + random.random()*self._RANDOM_NOISE - self._RANDOM_NOISE/2 , 4)
                #else:
                #    self._template[t_idx] = 0.0
            
    def get_next_label(self):
        self._current_label_position += 1
        if self._current_label_position < 0 or self._current_label_position >= len(self._RANDOM_LABEL_SEQUENCE):
            self._current_label_position = 0
        return self._RANDOM_LABEL_SEQUENCE[self._current_label_position]

    def get_next_sample_size(self):
        self._current_sample_size += 1
        if self._current_sample_size < self._MIN_SAMPLE_SIZE:
            self._current_sample_size = self._MIN_SAMPLE_SIZE
        if self._current_sample_size > self._MAX_SAMPLE_SIZE:
            self._current_sample_size = self._MIN_SAMPLE_SIZE
        return self._current_sample_size

    def add_new_instance(self,sample,label):
        instance_name = ''
        for t_idx in range(len(self._template)):
            if not str(self._template[t_idx]) == 'R':
                instance_name += 'F'+str(t_idx)+'V'+str(self._template[t_idx])
        instance_name += 'L'+label
        instance_name += 'S'+str(sample)
        self.fill_template_random_values()
        split = "train"
        if self._VERBOSE:
            print(instance_name,label,self._template) 
        if sample < self._TEST_SAMPLE_SIZE:
            split = "test"
        self._dataset[split]['sample'].append(instance_name)
        self._dataset[split]['label'].append(label)
        for f_idx in range(len(self._ATTS_VALUES)):
            self._dataset[split]['F'+str(f_idx)].append(self._template[f_idx])

    def json(self):
        return self._dataset["train"],self._dataset["test"]

    def dataframes(self):
        return pd.DataFrame.from_dict(self._dataset["train"]), pd.DataFrame.from_dict(self._dataset["test"]) 


    def generate(self):
        
        self.reset_dataset()
        v_rowid = 0
        
        if len(self._ATTS_VALUES) > self._NO_CATEG_ATTS + self._NO_CONTI_ATTS:
            raise Exception('[HXT:FAI:Err-101] Inconsistent _ATTS_VALUES.')

        self._current_label_position = -1
        self._current_sample_size = -1

        for attr1_idx in range(len(self._ATTS_VALUES)):
            for attr1_val in range(1,self._ATTS_VALUES[attr1_idx]+1):
                v_rowid += 1
                v_current_label = self.get_next_label()
                v_sample_size   = self.get_next_sample_size()
                if self._VERBOSE:
                    print(v_rowid,'Label',v_current_label,'SSize',v_sample_size,'Att1.idx',attr1_idx,"Att1.val",attr1_val) 
                for sample_id in range(v_sample_size):
                    self.reset_template()
                    self.set_template_value(attr1_idx,attr1_val)
                    if self._VERBOSE:
                        print('\t','Sample',sample_id,self.get_template()) 
                    self.add_new_instance(sample_id,v_current_label)

                for attr2_idx in range(attr1_idx+1,len(self._ATTS_VALUES)):
                    for attr2_val in range(1,self._ATTS_VALUES[attr2_idx]+1):
                        v_rowid += 1
                        v_current_label = self.get_next_label()
                        v_sample_size   = self.get_next_sample_size()
                        if self._VERBOSE:
                            print(v_rowid,'Label',v_current_label,'SSize',v_sample_size,'Att1.idx',attr1_idx,"Att1.val",attr1_val,'Att2.idx',attr2_idx,"Att2.val",attr2_val) 
                        for sample_id in range(v_sample_size):
                            self.reset_template()
                            self.set_template_value(attr1_idx,attr1_val)
                            self.set_template_value(attr2_idx,attr2_val)
                            if self._VERBOSE:
                                print('\t','Sample',sample_id,self.get_template()) 
                            self.add_new_instance(sample_id,v_current_label)

                        for attr3_idx in range(attr2_idx+1,len(self._ATTS_VALUES)):
                            for attr3_val in range(1,self._ATTS_VALUES[attr3_idx]+1):
                                v_rowid += 1
                                v_current_label = self.get_next_label()
                                v_sample_size   = self.get_next_sample_size()
                                if self._VERBOSE:
                                    print(v_rowid,'Label',v_current_label,'SSize',v_sample_size,'Att1.idx',attr1_idx,"Att1.val",attr1_val,'Att2.idx',attr2_idx,"Att2.val",attr2_val,'Att3.idx',attr3_idx,"Att3.val",attr3_val) 
                                for sample_id in range(v_sample_size):
                                    self.reset_template()
                                    self.set_template_value(attr1_idx,attr1_val)
                                    self.set_template_value(attr2_idx,attr2_val)
                                    self.set_template_value(attr3_idx,attr3_val)
                                    if self._VERBOSE:
                                        print('\t','Sample',sample_id,self.get_template()) 
                                    self.add_new_instance(sample_id,v_current_label)

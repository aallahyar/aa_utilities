class EntimICE:
    import pandas as pd

    # __version__ = '0.0.1'
    
    # initializations
    SCRATCH_DIR = '~/mounts/scratch/workspaces/course_dryrun/'
    STUDY_ID = 'D5241C00001'
    DATASET_NAMES = [
        'adsl', # Subject-Level Analysis Dataset
        'adlb', # Laboratory Test Results, Analysis Data
        'adre', # Respiratory Findings Analysis Dataset
    ]
    data = {}

    def __init__(self):
        self.load_data()
        self.sanity_checks()
    
    def __getitem__(self, dataset_name):
        return self.data[dataset_name]

    def load_data(self):
        # loading ADaM datasets
        for dn in EntimICE.DATASET_NAMES:
            self.data[dn] = pd.read_table(EntimICE.SCRATCH_DIR + f'{dn}.tsv.gz', index_col=0, low_memory=False) 
    
    def sanity_checks(self):
        # index and study checks
        for dn in EntimICE.DATASET_NAMES:
            self[dn].index.is_unique  # indices should be unique
            assert (self[dn]['STUDYID'] == EntimICE.STUDY_ID).all()  # all subjects should be from a single study

        # making sure all adre (respiratory) subjects are randomized
        assert self['adre']['RANDFL'].eq('Y').all()

        # sanity check: adsl and adlb should contain the same information
        adlb_subjects = list(set(self['adlb']['SUBJID']))
        common_cols = list(set(self['adsl'].columns) & set(self['adlb'].columns))
        df1 = self['adsl'][common_cols].query('SUBJID in @adlb_subjects').reset_index(drop=True)
        df2 = self['adlb'][common_cols].drop_duplicates().reset_index(drop=True)
        assert df1.equals(df2)


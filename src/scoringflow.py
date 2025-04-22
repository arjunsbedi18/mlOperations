from metaflow import FlowSpec, step, Flow

class WineClassifierScoringFlow(FlowSpec):

    @step
    def start(self):
        run = Flow('ClassifierTrainFlow').latest_run 
        self.train_run_id = run.pathspec 
        self.model = run['end'].task.data.model
        self.next(self.ingest_data)

    @step
    def ingest_data(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        X, y = datasets.load_wine(return_X_y=True)
        # Now using the test data, as we only used train and val before
        _, self.test_data, _, self.test_labels = train_test_split(X,y, test_size=0.2)
        print("Data loaded successfully")
        self.next(self.predict)

    @step
    def predict(self):
        self.predictions = self.model.predict(self.test_data)
        self.next(self.end)

    @step
    def end(self):
        print('Results')
        print('Model', self.model)
        print('Predictions')
        for i in range(5):
            print(self.test_labels[i], 'was predicted as ', self.predictions[i])

if __name__ == '__main__':
    WineClassifierScoringFlow()

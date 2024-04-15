import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp

class Mapper(api.Mapper):
    def map(self, context):
        row = context.value.strip().split(",")
        context.emit(row[1], 1)

class Reducer(api.Reducer):
    def reduce(self, context):
        context.emit(context.key, sum(context.values))

def __main__():
    pp.run_task(pp.Factory(Mapper, Reducer))

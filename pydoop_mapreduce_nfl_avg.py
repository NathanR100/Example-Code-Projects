import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp

class Mapper(api.Mapper):
    def map(self, context):
        row = context.value.strip().split(",")
        context.emit(row[1], row[25])

class Reducer(api.Reducer):
    def reduce(self, context):
        yards = context.values
        count = 0
        total_yards = 0
        for yard in yards:
            if yard != "NA":
                count += 1
                total_yards += int(yard)


        # Calculate average yards gained per play
        average_yards_per_play = total_yards / count if count > 0 else 0
        
        # Emit the result
        context.emit(context.key, str(average_yards_per_play))
        

def __main__():
    pp.run_task(pp.Factory(Mapper, Reducer))

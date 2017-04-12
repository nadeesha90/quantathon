from application.tasks.get_data import populate_stocks, populate_dailystats, populate_w2_mats
from application.tasks.optimize_part2 import optimize_part2
from application.tasks.optimize_part4 import optimize_part4
from application.tasks.plot_results import get_portfolio_and_plot_results




def load_all_data():
    # load csv and get stock-trade specific data
    populate_stocks()
    
    # load csv and get daily stats from stocks
    populate_dailystats()

    # load csv and get aggregate stat matrices
    populate_w2_mats()


if __name__ == "__main__":
    # First, setup database
    #load_all_data()

    # run optimizer for part 2
    #optimize_part2()

    # run optimizer for part 4 
    optimize_part4()
    
    # plot results
    #get_portfolio_and_plot_results()





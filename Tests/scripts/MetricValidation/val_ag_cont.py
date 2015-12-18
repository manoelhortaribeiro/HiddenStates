from val import *

x, x_t, y, y_t, t_x, t_x_t, t_y, t_y_t = load()

print "GREEDY:"

greedy_states, g_test = greedy(x, y, x_t, y_t, add=1, rangeof=6)

print "ARBITRARY:"

arbitrary_states, a_test = arbitrary(t_x, t_y, t_x_t, t_y_t, add=1, rangeof=6)

r_arb, r_gre = test(greedy_states, arbitrary_states, x, y, x_t, y_t, numberseeds=3)

print "Results:\n", "arb:", r_arb, "gre:", r_gre

path = os.environ['PYTHONPATH'].split(os.pathsep)[0]
f = open(path + "/Dataset/Output/Results/greedy", "w")
f.write("arb: " + str(r_arb))
f.write("\ngre: " + str(r_gre))
f.close()
import os
# os.system('rmdir /s /q test_results' )
# print("ok")
# ------------------

str = "./hello/"
str = str.replace("./", "").replace("/", "")
print(str)

# writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor是把sess.graph写进去？但是我保存的是
#     td_loss = tf.Variable(0.)
#     tf.summary.scalar("Beta", td_loss)
#     eps_total_reward = tf.Variable(0.)
#     tf.summary.scalar("Reward", eps_total_reward)
#     entropy = tf.Variable(0.)
#     tf.summary.scalar("Entropy", entropy)

#     summary_vars = [td_loss, eps_total_reward, entropy]
#     summary_ops = tf.summary.merge_all()
# 这些内容，请问会有影响吗？
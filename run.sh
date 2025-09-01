enqueue_compss \
  --sc_cfg=leonardo.cfg \
  --exec_time=10 \
  --num_nodes=2 \
  --lang=python \
  --master_working_dir=/leonardo/home/userexternal/tbarnie0/test/ST540103/mnt/tmp \
  --worker_working_dir=/leonardo/home/userexternal/tbarnie0/test/ST540103/mnt/tmp \
  --classpath=/leonardo/home/userexternal/tbarnie0/test/ST540103 \
  --pythonpath=/leonardo/home/userexternal/tbarnie0/test/ST540103 \
  --log_level=debug \
  --qos=boost_qos_dbg \
  --queue=boost_usr_prod \
  --cpus_per_node=8 \
  test.py

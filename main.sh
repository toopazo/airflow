
# database
# python -m airflow.database.db_create

# face_detector
# python -m airflow.face_detector.process_video \
#   airflow/videos airflow/output

# database
# python -m airflow.database.insert_video \
#     "videos/inauguracion_metro_santiago.mp4" \
#     "output/inauguracion_metro_santiago"

# face_sequencer
# python -m airflow.face_sequencer.find_sequences 1 output

# face_reider
python -m  airflow.face_reider.sequence_cluster_eval "output/inauguracion_metro_santiago"
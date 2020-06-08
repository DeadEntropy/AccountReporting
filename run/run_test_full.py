from process import process, status
from transforms import master_transform


mt = master_transform.Loader()
mt.load_save()

pr = process.Process()
pr.process_save()

st = status.LastUpdate()
st.last_update_save()

from Video_Resultado_2 import arquivo_sensor_fadiga
from pathlib import Path
import pandas as pd

entries = Path('Dataset/')
for entry in entries.iterdir():
	if entry.name.find(".")<0:
		entries_1 = Path(entry)
		for entry_1 in entries_1.iterdir():
			entries_2 = Path(entry_1)
			if entries_2.name.find(".DS_Store")<0:
				for entry_2 in entries_2.iterdir():
					if entry_2.name.find(".DS_Store")<0:
						entries_3 = Path(entry_2)
						for entry_3 in entries_3.iterdir():
							if entry_3.name.find(".DS_Store")<0:
								entries_4 = Path(entry_3)

<<<<<<< HEAD
								csv = pd.read_csv("results.csv", sep=',', encoding = "ISO-8859-1")
								print(str(entries_4), list(csv['arquivo']))
								if str(entries_4) not in list(csv['arquivo']):
									categoria, porcentagem = arquivo_sensor_fadiga(str(entries_4))
									print('Conclusão: {0} com {1:2f} %'.format(categoria, porcentagem))
=======
								csv = pd.read_csv("results.csv", usecols=[0,1,2], sep=';', encoding = "ISO-8859-1")
								csv = csv.dropna(axis=0, how='any') #retirando linhas NaN
								lista_nome_video = np.unique(csv['nome_video'])
								print(nome_video, lista_nome_video)
								if nome_video not in lista_nome_video:
									try: #ele lê uns arquivos estranhos como 55/._10.mp4 que nao devem ser considerados
										df = arquivo_features(str(entries_4), nome_video) 
										df.to_csv('results.csv', sep=';', index = False, mode='a', header=False)
									except Exception as e:
										print(e) 
										# traceback.print_exc()
									
>>>>>>> 12a8b3d6866fdebb58fe423e8077c51de987edd9

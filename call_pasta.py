from get_features import arquivo_features
from pathlib import Path
import pandas as pd
import numpy as np

pasta = '01'

entries = Path('D:/Rebeca/Dataset/')
for entry in entries.iterdir():
	if entry.name.find(".")<0:
		entries_1 = Path(entry)
		for entry_1 in entries_1.iterdir():
			entries_2 = Path(entry_1)
			if entries_2.name.find(".DS_Store")<0:
				for entry_2 in entries_2.iterdir():
                    if entry_2.name.find(".DS_Store")<0:
                        entries_3 = Path(entry_2)
                        if entries_3[-2:] == pasta:
                            for entry_3 in entries_3.iterdir():
                                if entry_3.name.find(".DS_Store")<0:
                                    entries_4 = Path(entry_3)
                                    palavras = str(entries_4).split('/')
                                    nome_video = palavras[-2]+'/'+ palavras[-1]

                                    csv = pd.read_csv("results.csv", usecols=[0,1,2], sep=';', encoding = "ISO-8859-1")
                                    csv = csv.dropna(axis=0, how='any') #retirando linhas NaN
                                    lista_nome_video = np.unique(csv['nome_video'])
                                    print(nome_video, lista_nome_video)
                                    if nome_video not in lista_nome_video:
                                        try: #ele lê uns arquivos estranhos como 55/._10.mp4 que nao devem ser considerados
                                            df = arquivo_features(str(entries_4), nome_video)
                                            df.query("ear!=-1" and "distancia_entre_os_labios!=-1", inplace=True) 
                                            print(df.shape)
                                            if df.shape[0]>200:
                                                df.to_csv('results.csv', sep=';', index = False, mode='a', header=False)
                                            else:
                                                print(nome_video)
                                        except Exception as e:
                                            print(e) 
                                            # traceback.print_exc()
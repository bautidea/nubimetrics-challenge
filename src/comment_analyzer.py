from typing import Literal
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
import re
import unicodedata
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import nltk
from nltk.corpus import stopwords


class Comment_Analyzer ():
    '''
    Esta clase se encarga de cargar, validar, limpiar, procesar y predecir comentarios.

    Atributos:
        path_to_comments (str): Ruta al archivo CSV que contiene los comentarios.
        vectorizer_path (str): Ruta al archivo del vectorizador entrenado.
        model_path (str): Ruta al archivo del modelo entrenado.
    '''

    def __init__(
        self,
        path_to_comments: str,
        vectorizer_path: str,
        model_path: str,
        output_path: str
    ):
        '''
            Inicializa la clase Comment_Analyzer.

            Args: 
                path_to_comments (str): Ruta al archivo CSV que contiene los comentarios.
                vectorizer_path (str): Ruta al archivo del vectorizador entrenado.
                model_path (str): Ruta al archivo del modelo entrenado.

            Se cargan, validan, limpian, preprocesan y predicen los datos.
            Ademas se realiza la carga del vectorizador y el modelo entrenado.
        '''
        # Asigno variables de instancia.
        self.path_to_comments = path_to_comments
        self.raw_data = self.__load_data()

        # Limpio la data cruda.
        self.stop_words = self.__load_stop_words()
        self.clean_data = self.__clean_data()

        # Cargo modelo y vectorizador para procesar la data y luego predecir.
        self.vectorizer_instance = self.__load_vectorizer(vectorizer_path)
        self.model_instance = self.__load_model(model_path)
        # Preprocesamos la data para luego predecir.
        self.processed_data = self.__preprocess_data()
        # Predecimos la data procesada.
        self.predictions = self.__predict_comments()

        # Guardamos resultados.
        self.__save_predictions(output_path)

    def __load_data(self):
        '''
            Carga los datos desde un archivo CSV y luego valida que los campos 'content' y 'title' existan.

            Devuelve:
                pd.DataFrame: con los datos validados.
        '''
        # Este metodo cargaria los datos, en este caso es un csv, pero cualquier tipo de extraccion
        # desde una consulta a SQL a archivos alojados en alguna nube.
        raw_data = pd.read_csv(self.path_to_comments, low_memory=False)
        raw_data = self.__validate_data(raw_data)
        return raw_data

    def __validate_data(self, raw_data: pd.DataFrame):
        '''
            Valida que el DF que se carga contenga los campos 'content' y 'title'.
            Ademas realiza limpieza de nulos si los hubiera, se tolera que el campo 'title' contenga nulos.
            Pero el campo 'content' no puede ser nulo, por esa razon se eliminan los registros con ese campo nulo.

            Args:
                raw_data (pd.DataFrame): DataFrame con los datos crudos.

            Devuelve:
                pd.DataFrame: con los datos validados y sin nulos en el campo 'content'.
        '''
        # Primero validaremos que el campo 'content' y 'title' esten presentes en el DF de entrada.
        if ('content' in raw_data.columns) and ('title' in raw_data.columns):
            # Si el campo 'content' contiene strings vacios, lo reemplazo por NaN.
            raw_data['content'] = raw_data['content'].replace(
                '', np.nan)
            # Si el campo 'title' contiene Nan lo reemplazo por string vacio, ya que no filtrare por este campo.
            raw_data['title'] = raw_data['title'].fillna('')

            print(
                f"Cantidad de registros sin contenido = {(raw_data['content'].isnull()).sum()}")
            print(
                f"Cantidad de registros sin titulo = {(raw_data['title'] == '').sum()}")

            # Dropeo comentarios con 'content' nulo.
            raw_data.dropna(subset=['content'], inplace=True)

            return raw_data
        else:
            raise Exception(
                'El campo content o title no se encuentra en el dataframe de entrada.')

    def __load_stop_words(self):
        '''
            Cargamos stopwords en español, para eliminar palabras que no aportan valor a la clasificacion.
        '''
        print('Cargando stop words...')
        nltk.download('stopwords')
        return set(stopwords.words('spanish'))

    def __clean_data(self):
        '''
            Limpia y normaliza el texto en los comentarios, aplicando:
                - Transformacion de texto a minuscula.
                - Eliminacion de numeros, caracteres especiales, signos de puntuacion, 
                    espacios multiples y stopwords.
                - Tokenizacion y stemming.

            Devuelve:
                pd.DataFrame: con los datos limpios y normalizados.
        '''

        # Funcion interna para realizar Stemming.
        # Aaplicara Stemming a los commentarios, para reducir las palabras que componen a los
        # comentarios a su raiz etimologica.
        def stemming(text: str):
            stemmer = SnowballStemmer(language='spanish')
            # Stemming. Transformamos texto a lista ya que el stemmer trabaja sobre string crudos, no sobre
            # linguisticos como lemmatization.
            tokens = text.split()

            return [stemmer.stem(token) for token in tokens if token not in self.stop_words]

        # Funcion interna para realizar limpieza de texto.
        def clean_text(text: str):
            # Transformo texto a minusculas.
            text = str(text).lower()

            # Elimino tildes/acentos.
            text = unicodedata.normalize('NFKD', text).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            # Elimino numeros.
            text = re.sub(r'\d+', ' ', text)
            # Elimino signos de puntuacion.
            text = re.sub(r'[^\w\s]', ' ', text)
            # Hay algunos comentarios que tienen la nueva linea como '\n' especificamente
            # reemplazare '\n' -> ' '
            text = re.sub(r'\n', ' ', text)
            # Eliminamos espacios multiples.
            text = re.sub(r'\s+', ' ', text)

            # Stemmingzamos.
            tokens = stemming(text)

            text = ' '.join(tokens)

            # Vuelvo a eliminar tildes por si Lemmatizacion introdujo alguna.
            text = unicodedata.normalize('NFKD', text).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            return text

        # Para no tocar la referencia original, se crea una copia del dataframe.
        clean_data = self.raw_data.copy()

        # Una vez validados los dos campos, procedemos a crear el campo 'text_label' que contendra
        # el texto limpio y procesado.
        clean_data['text_label'] = clean_data['title'] + \
            ' ' + clean_data['content']

        # Aplicamos la limpieza de texto a cada comentario.
        clean_data['text_label'] = clean_data['text_label'].apply(clean_text)
        return clean_data

    def __load_model(self, model_path: str):
        '''
            Carga el modelo de clasificacion entrenado previamente en la notebook de resolucion.

            Args:
                model_path (str): Ruta al archivo del modelo entrenado.

            Devuelve:
                object: Instancia del modelo.
        '''
        try:
            instance = joblib.load(model_path)
            print('Modelo cargado correctamente...')
            print(instance)
            return instance
        except Exception as e:
            raise Exception(
                f"No se pudo cargar el modelo desde '{model_path}'. Error: {e}")

    def __load_vectorizer(self, vectorizer_path: str):
        '''
            Carga el vectorizador entrenado previamente en la notebook de resolucion.

            Args:
                vectorizer_path (str): Ruta al archivo del vectorizador entrenado.

            Devuelve:
                object: Instancia del vectorizador.
        '''
        try:
            instance = joblib.load(vectorizer_path)
            print(f'Vectorizador cargado correctamente...')
            return instance
        except Exception as e:
            raise Exception(
                f"No se pudo cargar el vectorizador desde '{vectorizer_path}'. Error: {e}")

    def __preprocess_data(self):
        '''
            Preprocesa los datos aplicando encoding con el vectorizador entrenado. Toma el atributo del DF
            limpio y aplica el vectorizador a la columna 'text_label'.

            Devuelve:
                pd.DataFrame: con los datos preprocesados y codificados.
        '''
        # Este metodo preprocesara los datos, aplicando el vectorizador a los comentarios limpios.
        processed_data = self.clean_data.copy()

        word_count = self.vectorizer_instance.transform(
            processed_data['text_label']
        ).toarray()

        return pd.DataFrame(
            word_count, columns=self.vectorizer_instance.get_feature_names_out()
        )

    def __predict_comments(self):
        '''
            Predice si un comentario es:
                - Negativo: 0
                - Neutro: 1
                - Positivo: 2

            Utilizando el atributo 'processed_data' que contiene los datos codificados y 'model_instance' que contiene 
            la instancia del modelo entrenado.

            Devuelve:
                np.ndarray: con las predicciones de los comentarios.
        '''
        return self.model_instance.predict(self.processed_data)

    def __save_predictions(self, output_path):
        '''
            Guarda las predicciones sobre la data cruda en el path declarado.

            Args:
                output_path (str): Path de donde se espera el output.

            Devuelve:
                None: Se genera un .csv en el path declarado.
        '''
        # Creo copya de la instancia para no trabajar sobre la instancia.
        output_df = self.raw_data.copy()
        output_df['predictions'] = self.predictions
        return output_df.to_csv(output_path, index=False)

    def return_word_cloud(self, on_data: Literal['raw', 'clean', 'predicted'] = 'predicted'):
        '''
            Genera una WordCloud para diferentes instancias de los datos:
                - raw: Data cruda.
                - clean: Data limpia.
                - predicted: Data procesada (codificada) y predicha.

            La funcion de este metodo es poder visualizar como estan compuestos los comentarios, para tratar de hacer
            un analisis de los comentarios e interpretar las predicciones de manera simple.

            Para el caso de las predicciones, se genera una WordCloud que segmenta las nubes en base a las predicciones.

            Args:
                on_data (str): Tipo de datos a visualizar. Puede ser 'raw', 'clean' o 'predicted'.

            Devuelve:
                None: Un plot de la WordCloud generada para cada instancia de los datos.
        '''

        # Funcion interna para generar la WordCloud.
        # Se le pasa el texto a graficar y el titulo de la grafica.
        def plot_word_cloud(text, title, generate_form_frequecies=False):
            # generate_from_frequencies se utiliza cuando cada columna representa una palabra, es decir
            # cuando el texto ya esta codificado o procesado, caso contrario se utiliza todo el texto en un comentario
            # que es cuando la data esta cruda o limpia (sin stopwords, ni ruido)
            if not generate_form_frequecies:
                wordcloud = WordCloud(width=900, height=900).generate(text)
            else:
                wordcloud = WordCloud(
                    width=1000, height=1000).generate_from_frequencies(text)

            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"WordCloud -  {title}")
            plt.show()

        # Si se quiere visualizar nube sobre la data cruda.
        if on_data == 'raw':
            # Concatenamos todos los comentarios en un unico string.
            text = ' '.join(
                self.raw_data['content'].str.lower().astype(str)
            )
            plot_word_cloud(text, 'Data Cruda')

        # Si se quiere visualizar nube sobre la data limpia.
        elif on_data == 'clean':
            text = ' '.join(
                self.clean_data['text_label'].astype(str)
            )
            plot_word_cloud(text, 'Data Limpia')

        # Si se quiere visualizar sobre la data procesada y predicha.
        elif on_data == 'predicted':
            # Creo copia de DF para poder discretizar entre las distintas clases.
            predicted_data_df = self.processed_data.copy()
            # Creo columna de prediciones.
            predicted_data_df['target'] = self.predictions
            # Lista de valores unicos de clases [0,1,2].
            target_list = predicted_data_df['target'].sort_values(
            ).unique().tolist()

            # Por cada valor en la lista de valores [0,1,2]
            for target_iter in target_list:
                # Selecciono la clase en el loop.
                mask = predicted_data_df['target'] == target_iter
                # Me quedo unicamente con la clase del loop.
                df_iter = predicted_data_df[mask].copy()
                # De las columnas (cada columna es una palabra o ngram) obtengo la frequencia
                # total de aparicion,
                word_freq = df_iter.drop(
                    columns=['target']).sum(axis=0).to_dict()

                plot_word_cloud(
                    word_freq, f'Clase: {target_iter}', generate_form_frequecies=True)
        else:
            raise Exception(
                'El parametro "on_data" debe ser "raw", "clean" o "predicted".')

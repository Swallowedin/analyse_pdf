import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
import pdf2image
import tabula
import tempfile
import logging
import os
import base64
import io

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Extracteur de Tableaux PDF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pdf-table-extractor')

class PDFTableExtractor:
    """
    Classe pour extraire des tableaux à partir de fichiers PDF, 
    en utilisant OCR pour les PDF qui contiennent des tableaux sous forme d'images.
    """
    
    def __init__(self, pdf_file, output_format='csv'):
        """
        Initialise l'extracteur de tableaux PDF.
        
        Args:
            pdf_file (BytesIO): Fichier PDF chargé en mémoire.
            output_format (str, optional): Format de sortie ('csv', 'xlsx', ou 'json').
        """
        self.pdf_file = pdf_file
        self.output_format = output_format.lower()
        
        # Créer un dossier temporaire pour les fichiers intermédiaires
        self.temp_dir = tempfile.mkdtemp()
        
        # Vérifier le format de sortie
        if self.output_format not in ['csv', 'xlsx', 'json']:
            raise ValueError("Le format de sortie doit être 'csv', 'xlsx' ou 'json'.")
    
    def extract_tables_direct(self):
        """
        Extraction directe des tableaux à partir d'un PDF en utilisant tabula-py.
        
        Returns:
            list: Liste des DataFrames pandas représentant les tableaux extraits.
        """
        logger.info("Tentative d'extraction directe...")
        
        try:
            # Sauvegarder le fichier temporairement pour tabula
            temp_pdf_path = os.path.join(self.temp_dir, "temp.pdf")
            with open(temp_pdf_path, 'wb') as f:
                f.write(self.pdf_file.getvalue())
            
            # Extraire tous les tableaux du PDF
            tables = tabula.read_pdf(
                temp_pdf_path, 
                pages='all', 
                multiple_tables=True,
                guess=True
            )
            
            if tables and len(tables) > 0:
                logger.info(f"Extraction directe réussie. {len(tables)} tableaux trouvés.")
                return tables
            else:
                logger.info("Aucun tableau n'a été détecté par extraction directe.")
                return []
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction directe: {str(e)}")
            return []
    
    def extract_tables_ocr(self):
        """
        Extraction des tableaux à l'aide de l'OCR lorsque l'extraction directe ne fonctionne pas.
        
        Returns:
            list: Liste des DataFrames pandas représentant les tableaux extraits.
        """
        logger.info("Tentative d'extraction par OCR...")
        
        tables = []
        
        try:
            # Convertir le PDF en images
            images = pdf2image.convert_from_bytes(self.pdf_file.getvalue())
            
            for i, img in enumerate(images):
                logger.info(f"Traitement de la page {i+1}...")
                
                # Sauvegarder l'image temporairement
                img_path = os.path.join(self.temp_dir, f"page_{i+1}.png")
                img.save(img_path, 'PNG')
                
                # Détecter et extraire les tableaux de l'image
                tables_from_image = self.detect_tables_from_image(img_path, i+1)
                if tables_from_image:
                    tables.extend(tables_from_image)
            
            if tables:
                logger.info(f"Extraction OCR réussie. {len(tables)} tableaux trouvés.")
            else:
                logger.warning("Aucun tableau n'a été détecté par OCR.")
                
            return tables
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction OCR: {str(e)}")
            return []
    
    def detect_tables_from_image(self, img_path, page_num):
        """
        Détecte et extrait les tableaux d'une image en utilisant des techniques de vision par ordinateur.
        
        Args:
            img_path (str): Chemin vers l'image à traiter.
            page_num (int): Numéro de la page.
            
        Returns:
            list: Liste des DataFrames extraits de l'image.
        """
        # Charger l'image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarisation et filtrage
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Détection des lignes horizontales et verticales
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combinaison des lignes horizontales et verticales
        table_mask = horizontal_lines + vertical_lines
        
        # Trouver les contours des tableaux potentiels
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrer les petits contours (qui ne sont probablement pas des tableaux)
            if w < 100 or h < 100:
                continue
                
            # Extraire la région du tableau
            table_roi = img[y:y+h, x:x+w]
            
            # Sauvegarder la région dans un fichier temporaire
            temp_table_path = os.path.join(self.temp_dir, f"table_p{page_num}_t{i+1}.png")
            cv2.imwrite(temp_table_path, table_roi)
            
            # Utiliser OCR pour extraire le texte du tableau
            try:
                # Utiliser pytesseract pour OCR
                ocr_text = pytesseract.image_to_string(Image.open(temp_table_path))
                
                # Utiliser pytesseract spécifiquement pour les tableaux
                ocr_data = pytesseract.image_to_data(Image.open(temp_table_path), output_type=pytesseract.Output.DATAFRAME)
                
                # Tenter de reconstruire une structure tabulaire
                table_data = self.reconstruct_table_from_ocr(ocr_data, ocr_text)
                
                if not table_data.empty:
                    tables.append(table_data)
                    logger.info(f"Tableau {i+1} extrait de la page {page_num}")
            
            except Exception as e:
                logger.error(f"Erreur lors de l'OCR du tableau {i+1} de la page {page_num}: {str(e)}")
        
        return tables
    
    def reconstruct_table_from_ocr(self, ocr_data, ocr_text):
        """
        Reconstruit une structure tabulaire à partir des données OCR.
        
        Args:
            ocr_data (DataFrame): Données OCR structurées.
            ocr_text (str): Texte brut extrait par OCR.
            
        Returns:
            DataFrame: Tableau reconstruit sous forme de DataFrame.
        """
        try:
            # Nettoyage des données OCR
            ocr_data = ocr_data[ocr_data['conf'] > 30]  # Filtrer les détections de faible confiance
            
            # Identifier les lignes du tableau en regroupant par coordonnées y
            ocr_data['line_num'] = ocr_data['top'] // 10  # Regroupement approximatif par ligne
            
            # Identifier les colonnes du tableau en regroupant par coordonnées x
            unique_lefts = sorted(ocr_data['left'].unique())
            
            # Simplifier en trouvant les centres de colonnes principales
            if len(unique_lefts) > 1:
                # Algorithme de clustering simple pour identifier les colonnes
                col_clusters = []
                current_cluster = [unique_lefts[0]]
                
                for i in range(1, len(unique_lefts)):
                    if unique_lefts[i] - unique_lefts[i-1] < 50:  # Seuil de proximité
                        current_cluster.append(unique_lefts[i])
                    else:
                        col_clusters.append(current_cluster)
                        current_cluster = [unique_lefts[i]]
                
                if current_cluster:
                    col_clusters.append(current_cluster)
                
                col_centers = [sum(cluster) / len(cluster) for cluster in col_clusters]
                
                # Associer chaque élément à une colonne
                def assign_column(x):
                    dists = [abs(x - center) for center in col_centers]
                    return dists.index(min(dists))
                
                ocr_data['col_num'] = ocr_data['left'].apply(assign_column)
            else:
                ocr_data['col_num'] = 0
            
            # Construire le tableau
            table_data = pd.DataFrame()
            
            # Trouver les en-têtes (première ligne, si applicable)
            headers = ocr_data[ocr_data['line_num'] == ocr_data['line_num'].min()]
            
            for line_num in sorted(ocr_data['line_num'].unique()):
                row_data = {}
                line_items = ocr_data[ocr_data['line_num'] == line_num]
                
                for _, item in line_items.iterrows():
                    col_key = headers.iloc[item['col_num']]['text'] if len(headers) > item['col_num'] else f"Col_{item['col_num']}"
                    row_data[col_key] = item['text']
                
                table_data = pd.concat([table_data, pd.DataFrame([row_data])], ignore_index=True)
            
            return table_data
        
        except Exception as e:
            logger.error(f"Erreur lors de la reconstruction du tableau: {str(e)}")
            
            # Fallback: créer un DataFrame simple à partir du texte brut
            lines = ocr_text.strip().split('\n')
            if not lines:
                return pd.DataFrame()
            
            # Essayer de détecter un séparateur de colonnes (espace, tabulation, etc.)
            if '\t' in lines[0]:
                separator = '\t'
            else:
                # Utiliser des espaces multiples comme séparateur
                separator = r'\s{2,}'
            
            rows = []
            for line in lines:
                if line.strip():
                    rows.append(line.split(separator))
            
            if not rows:
                return pd.DataFrame()
            
            # Normaliser la longueur des lignes
            max_cols = max(len(row) for row in rows)
            normalized_rows = [row + [''] * (max_cols - len(row)) for row in rows]
            
            # Créer le DataFrame
            return pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0] if len(normalized_rows) > 1 else None)
    
    def process(self):
        """
        Traite le PDF et extrait les tableaux, en utilisant d'abord l'extraction directe,
        puis l'OCR si nécessaire.
        
        Returns:
            list: Liste des DataFrames des tableaux extraits.
        """
        # Tenter l'extraction directe
        tables = self.extract_tables_direct()
        
        # Si aucun tableau n'est trouvé, essayer l'OCR
        if not tables:
            tables = self.extract_tables_ocr()
        
        # Nettoyer les fichiers temporaires
        self.cleanup()
        
        return tables
    
    def cleanup(self):
        """Nettoie les fichiers temporaires."""
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {str(e)}")


def get_table_download_link(df, filename, format_type):
    """
    Génère un lien de téléchargement pour un DataFrame.
    
    Args:
        df (DataFrame): Le DataFrame à télécharger.
        filename (str): Nom du fichier.
        format_type (str): Format du fichier ('csv', 'xlsx', 'json').
        
    Returns:
        str: HTML pour le lien de téléchargement.
    """
    if format_type == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        mime_type = 'text/csv'
        file_extension = 'csv'
    elif format_type == 'xlsx':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Tableau', index=False)
        b64 = base64.b64encode(output.getvalue()).decode()
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_extension = 'xlsx'
    elif format_type == 'json':
        json_str = df.to_json(orient='records', indent=4)
        b64 = base64.b64encode(json_str.encode()).decode()
        mime_type = 'application/json'
        file_extension = 'json'
    else:
        return ""

    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{file_extension}">Télécharger {filename}.{file_extension}</a>'
    return href


def main():
    st.title("Extracteur de Tableaux PDF")
    st.markdown("""
    Cette application vous permet d'extraire des tableaux à partir de fichiers PDF, 
    même lorsque ces tableaux sont sous forme d'images. L'application utilise deux approches:
    
    1. **Extraction directe** pour les PDF contenant des tableaux structurés
    2. **Extraction par OCR** pour les PDF contenant des tableaux sous forme d'images
    """)
    
    # Sidebar
    st.sidebar.title("Options")
    output_format = st.sidebar.selectbox(
        "Format de sortie",
        options=["csv", "xlsx", "json"],
        index=0
    )
    
    # Téléchargement de fichier
    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=['pdf'])
    
    if uploaded_file is not None:
        # Afficher les informations du fichier
        file_details = {
            "Nom du fichier": uploaded_file.name,
            "Taille": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("**Détails du fichier:**")
        for key, value in file_details.items():
            st.write(f"{key}: {value}")
        
        # Bouton pour lancer l'extraction
        if st.button("Extraire les tableaux"):
            with st.spinner('Extraction en cours... Cela peut prendre quelques minutes.'):
                try:
                    # Initialiser l'extracteur
                    extractor = PDFTableExtractor(uploaded_file, output_format)
                    
                    # Extraire les tableaux
                    tables = extractor.process()
                    
                    if tables and len(tables) > 0:
                        st.success(f"{len(tables)} tableaux ont été extraits avec succès!")
                        
                        # Afficher et permettre le téléchargement de chaque tableau
                        for i, table in enumerate(tables):
                            if not table.empty:
                                st.subheader(f"Tableau {i+1}")
                                
                                # Afficher le tableau
                                st.dataframe(table)
                                
                                # Lien de téléchargement
                                table_filename = f"{uploaded_file.name.split('.')[0]}_table_{i+1}"
                                download_link = get_table_download_link(table, table_filename, output_format)
                                st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.warning("Aucun tableau n'a pu être extrait du document.")
                
                except Exception as e:
                    st.error(f"Une erreur s'est produite: {str(e)}")
    
    # Informations supplémentaires
    st.markdown("---")
    st.subheader("Comment ça marche?")
    st.markdown("""
    1. **Téléchargez** un fichier PDF contenant des tableaux
    2. Cliquez sur **Extraire les tableaux** pour lancer le processus
    3. Visualisez les tableaux extraits et **téléchargez-les** dans le format de votre choix
    
    L'application tente d'abord une extraction directe des tableaux. Si cela échoue, elle utilise l'OCR 
    (reconnaissance optique de caractères) pour extraire le texte des images de tableaux.
    
    **Note:** La qualité de l'extraction dépend fortement de la qualité du PDF et de la structure des tableaux.
    """)


if __name__ == "__main__":
    main()

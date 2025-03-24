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
import re
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Extracteur de Relevés de Charges",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pdf-charges-extractor')

class PDFChargesExtractor:
    """
    Classe pour extraire des tableaux de charges à partir de fichiers PDF,
    en utilisant OCR si nécessaire et en structurant les données pour analyse.
    """
    
    def __init__(self, pdf_file):
        """
        Initialise l'extracteur de relevés de charges.
        
        Args:
            pdf_file (BytesIO): Fichier PDF chargé en mémoire.
        """
        self.pdf_file = pdf_file
        
        # Créer un dossier temporaire pour les fichiers intermédiaires
        self.temp_dir = tempfile.mkdtemp()
        
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
                guess=True,
                lattice=True,
                stream=True
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
                ocr_text = pytesseract.image_to_string(Image.open(temp_table_path), lang='fra')
                
                # Configuration spéciale pour les tableaux
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                ocr_data = pytesseract.image_to_data(
                    Image.open(temp_table_path), 
                    output_type=pytesseract.Output.DATAFRAME,
                    config=custom_config,
                    lang='fra'
                )
                
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
                    rows.append(re.split(separator, line))
            
            if not rows:
                return pd.DataFrame()
            
            # Normaliser la longueur des lignes
            max_cols = max(len(row) for row in rows)
            normalized_rows = [row + [''] * (max_cols - len(row)) for row in rows]
            
            # Créer le DataFrame
            return pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0] if len(normalized_rows) > 1 else None)
    
    def process_charges_data(self, tables):
        """
        Traite les tableaux extraits pour identifier et structurer les données de charges.
        
        Args:
            tables (list): Liste des DataFrames pandas contenant les tableaux extraits.
            
        Returns:
            DataFrame: DataFrame structuré des charges.
        """
        # Initialiser un DataFrame pour stocker les données de charges
        charges_data = pd.DataFrame(columns=[
            'Code', 'Désignation', 'Montant HT', 'Montant TVA', 'Tantièmes globaux', 
            'Tantièmes particuliers', 'Quote-part', 'Pourcentage'
        ])
        
        # Identifier les tableaux de charges
        for i, table in enumerate(tables):
            # Nettoyer les noms de colonnes
            if table.columns.dtype == 'object':
                table.columns = [str(col).strip() for col in table.columns]
            
            # Détecter si c'est un tableau de charges
            charges_columns = [col for col in table.columns if 'désignation' in str(col).lower() or 
                              'mont' in str(col).lower() or 'tva' in str(col).lower() or 
                              'code' in str(col).lower() or 'chapitre' in str(col).lower()]
            
            if len(charges_columns) >= 2:
                logger.info(f"Tableau {i+1} identifié comme tableau de charges")
                
                # Nettoyer et préparer le tableau
                cleaned_table = self.clean_charges_table(table)
                if not cleaned_table.empty:
                    charges_data = pd.concat([charges_data, cleaned_table], ignore_index=True)
        
        # Si aucun tableau de charges n'a été trouvé, essayer une approche plus générique
        if charges_data.empty and tables:
            # Essayer de détecter des modèles dans les tableaux
            for i, table in enumerate(tables):
                if len(table.columns) >= 3 and len(table) > 2:
                    # Essayer d'identifier les colonnes par leur contenu
                    numeric_cols = []
                    text_cols = []
                    
                    for col in table.columns:
                        # Vérifier si la colonne contient principalement des valeurs numériques
                        try:
                            numeric_values = pd.to_numeric(table[col], errors='coerce')
                            if numeric_values.notna().sum() / len(table) > 0.5:
                                numeric_cols.append(col)
                            else:
                                text_cols.append(col)
                        except:
                            text_cols.append(col)
                    
                    if numeric_cols and text_cols:
                        logger.info(f"Tableau {i+1} traité de manière générique")
                        
                        # Renommer les colonnes de manière générique
                        renamed_table = table.copy()
                        rename_map = {}
                        
                        for i, col in enumerate(text_cols):
                            if i == 0:
                                rename_map[col] = 'Code'
                            elif i == 1:
                                rename_map[col] = 'Désignation'
                            else:
                                rename_map[col] = f'Texte_{i}'
                        
                        for i, col in enumerate(numeric_cols):
                            if i == 0:
                                rename_map[col] = 'Montant HT'
                            elif i == 1:
                                rename_map[col] = 'Montant TVA'
                            elif i == 2:
                                rename_map[col] = 'Quote-part'
                            else:
                                rename_map[col] = f'Valeur_{i}'
                        
                        renamed_table = renamed_table.rename(columns=rename_map)
                        charges_data = pd.concat([charges_data, renamed_table], ignore_index=True)
        
        # Nettoyer et formater les données
        if not charges_data.empty:
            # Convertir les colonnes numériques
            numeric_columns = ['Montant HT', 'Montant TVA', 'Quote-part']
            for col in numeric_columns:
                if col in charges_data.columns:
                    charges_data[col] = self.convert_to_numeric(charges_data[col])
            
            # Ajouter une colonne de pourcentage si possible
            if 'Montant HT' in charges_data.columns and 'Quote-part' in charges_data.columns:
                charges_data['Pourcentage'] = charges_data.apply(
                    lambda row: row['Quote-part'] / row['Montant HT'] * 100 if row['Montant HT'] != 0 else 0, 
                    axis=1
                )
                charges_data['Pourcentage'] = charges_data['Pourcentage'].round(2)
        
        return charges_data
    
    def clean_charges_table(self, table):
        """
        Nettoie et structure un tableau de charges.
        
        Args:
            table (DataFrame): Tableau de charges brut.
            
        Returns:
            DataFrame: Tableau de charges nettoyé et structuré.
        """
        try:
            # Copie du tableau pour éviter de modifier l'original
            df = table.copy()
            
            # Nettoyer les noms de colonnes
            if df.columns.dtype == 'object':
                column_mapping = {}
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'code' in col_lower or 'chapitre' in col_lower or 'clé' in col_lower:
                        column_mapping[col] = 'Code'
                    elif 'désignation' in col_lower or 'libellé' in col_lower or 'intitulé' in col_lower:
                        column_mapping[col] = 'Désignation'
                    elif 'mont' in col_lower and 'ht' in col_lower:
                        column_mapping[col] = 'Montant HT'
                    elif 'mont' in col_lower and 'tva' in col_lower:
                        column_mapping[col] = 'Montant TVA'
                    elif 'mont' in col_lower and 'ttc' in col_lower:
                        column_mapping[col] = 'Montant TTC'
                    elif 'tantième' in col_lower and ('globaux' in col_lower or 'total' in col_lower):
                        column_mapping[col] = 'Tantièmes globaux'
                    elif 'tantième' in col_lower and 'particulier' in col_lower:
                        column_mapping[col] = 'Tantièmes particuliers'
                    elif 'quote' in col_lower and 'part' in col_lower:
                        column_mapping[col] = 'Quote-part'
                    elif 'répartir' in col_lower and 'ht' in col_lower:
                        column_mapping[col] = 'Quote-part'
                
                # Renommer les colonnes identifiées
                if column_mapping:
                    df = df.rename(columns=column_mapping)
            
            # Supprimer les lignes vides ou ne contenant que des NaN
            df = df.dropna(how='all')
            
            # Nettoyer les valeurs numériques
            numeric_columns = ['Montant HT', 'Montant TVA', 'Montant TTC', 'Quote-part', 
                              'Tantièmes globaux', 'Tantièmes particuliers']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = self.convert_to_numeric(df[col])
            
            # Filtrer les lignes qui ne contiennent pas de données de charges valides
            if 'Désignation' in df.columns and 'Montant HT' in df.columns:
                df = df[df['Désignation'].notna() & df['Montant HT'].notna()]
            
            return df
        
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du tableau de charges: {str(e)}")
            return pd.DataFrame()
    
    def convert_to_numeric(self, series):
        """
        Convertit une série en valeurs numériques en gérant les formats spécifiques.
        
        Args:
            series (Series): Série à convertir.
            
        Returns:
            Series: Série convertie en numérique.
        """
        if series.dtype == 'object':
            # Fonction pour nettoyer les chaînes de caractères
            def clean_numeric_string(s):
                if pd.isna(s):
                    return np.nan
                
                if isinstance(s, (int, float)):
                    return float(s)
                
                s = str(s)
                # Supprimer les espaces, remplacer les virgules par des points
                s = s.replace(' ', '').replace(',', '.').replace('€', '')
                
                # Essayer de convertir en float
                try:
                    return float(s)
                except ValueError:
                    return np.nan
            
            return series.apply(clean_numeric_string)
        
        return series
    
    def extract_metadata(self):
        """
        Extrait les métadonnées du document (société, immeuble, période, etc.).
        
        Returns:
            dict: Dictionnaire des métadonnées extraites.
        """
        metadata = {
            'société': None,
            'immeuble': None,
            'adresse': None,
            'période': None,
            'type_document': None
        }
        
        try:
            # Sauvegarder le fichier temporairement
            temp_pdf_path = os.path.join(self.temp_dir, "temp.pdf")
            with open(temp_pdf_path, 'wb') as f:
                f.write(self.pdf_file.getvalue())
            
            # Extraire le texte de la première page
            images = pdf2image.convert_from_path(temp_pdf_path, first_page=1, last_page=1)
            if images:
                img_path = os.path.join(self.temp_dir, "first_page.png")
                images[0].save(img_path, 'PNG')
                
                # OCR sur la première page
                text = pytesseract.image_to_string(Image.open(img_path), lang='fra')
                
                # Extraire les métadonnées avec des expressions régulières
                # Société
                societe_match = re.search(r'Société\s*:?\s*([A-Z0-9 ]+)', text, re.IGNORECASE)
                if societe_match:
                    metadata['société'] = societe_match.group(1).strip()
                
                # Immeuble
                immeuble_match = re.search(r'Immeuble\s*:?\s*([A-Z0-9 ]+)', text, re.IGNORECASE)
                if immeuble_match:
                    metadata['immeuble'] = immeuble_match.group(1).strip()
                
                # Adresse (recherche du code postal et ville)
                adresse_match = re.search(r'\b(\d{5})\s+([A-Z]+)\b', text)
                if adresse_match:
                    # Rechercher une ligne d'adresse autour du code postal
                    cp_index = text.find(adresse_match.group(0))
                    line_start = text.rfind('\n', 0, cp_index) + 1
                    line_end = text.find('\n', cp_index)
                    
                    if line_start > 0 and line_end > cp_index:
                        metadata['adresse'] = text[line_start:line_end].strip()
                
                # Période
                periode_match = re.search(r'[Pp]ériode\s+du\s+(\d{2}/\d{2}/\d{4})\s+au\s+(\d{2}/\d{2}/\d{4})', text)
                if periode_match:
                    metadata['période'] = f"{periode_match.group(1)} au {periode_match.group(2)}"
                
                # Type de document
                if 'RELEVE GENERAL DE DEPENSES' in text:
                    metadata['type_document'] = 'Relevé général de dépenses'
                elif 'RELEVE INDIVIDUEL DES CHARGES' in text:
                    metadata['type_document'] = 'Relevé individuel des charges locatives'
                elif 'FACTURE' in text:
                    metadata['type_document'] = 'Facture'
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des métadonnées: {str(e)}")
        
        return metadata
    
    def process(self):
        """
        Traite le PDF et extrait les tableaux et les données de charges.
        
        Returns:
            tuple: (DataFrame des charges, dict des métadonnées)
        """
        # Extraire les métadonnées
        metadata = self.extract_metadata()
        
        # Tenter l'extraction directe
        tables = self.extract_tables_direct()
        
        # Si aucun tableau n'est trouvé, essayer l'OCR
        if not tables:
            logger.info("Aucun tableau trouvé par extraction directe. Tentative par OCR...")
            tables = self.extract_tables_ocr()
        
        # Traiter les tableaux pour en extraire les données de charges
        charges_data = self.process_charges_data(tables)
        
        # Nettoyer les fichiers temporaires
        self.cleanup()
        
        return charges_data, metadata
    
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
        csv = df.to_csv(index=False, sep=';')
        b64 = base64.b64encode(csv.encode()).decode()
        mime_type = 'text/csv'
        file_extension = 'csv'
    elif format_type == 'xlsx':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Charges', index=False)
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

    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{file_extension}" class="download-link">Télécharger {filename}.{file_extension}</a>'
    return href


def main():
    st.title("Extracteur de Relevés de Charges")
    st.markdown("""
    Cette application vous permet d'extraire automatiquement les données de charges à partir de relevés PDF de copropriété
    ou de charges locatives, structurant les informations dans un format analysable pour validation des clauses contractuelles de bail.
    
    #### Fonctionnalités :
    - Extraction des tableaux de charges depuis les PDFs (même basés sur des images)
    - Identification automatique des différentes charges et quotes-parts
    - Structuration des données pour analyse comparative
    - Export en CSV, Excel ou JSON pour analyse approfondie
    - Visualisation graphique des principales charges
    """)
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .download-link {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .metadata-box {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .charges-info {
            font-size: 18px;
            font-weight: bold;
            margin: 15px 0;
        }
        .highlight-row {
            background-color: #e6f7ff;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Options")
    output_format = st.sidebar.selectbox(
        "Format de téléchargement",
        options=["csv", "xlsx", "json"],
        index=0
    )
    
    # Zone de téléchargement de fichier principal
    uploaded_file = st.file_uploader("Choisissez un relevé de charges au format PDF", type=['pdf'])
    
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
        if st.button("Extraire les données de charges"):
            with st.spinner('Extraction en cours... Cela peut prendre quelques minutes.'):
                try:
                    # Initialiser l'extracteur
                    extractor = PDFChargesExtractor(uploaded_file)
                    
                    # Extraire les données de charges
                    charges_data, metadata = extractor.process()
                    
                    if not charges_data.empty:
                        # Afficher les métadonnées extraites
                        st.subheader("Informations du document")
                        metadata_html = "<div class='metadata-box'>"
                        for key, value in metadata.items():
                            if value:
                                metadata_html += f"<p><strong>{key.capitalize()}:</strong> {value}</p>"
                        metadata_html += "</div>"
                        st.markdown(metadata_html, unsafe_allow_html=True)
                        
                        # Afficher les statistiques des charges
                        if 'Montant HT' in charges_data.columns and 'Quote-part' in charges_data.columns:
                            total_ht = charges_data['Montant HT'].sum()
                            total_quote_part = charges_data['Quote-part'].sum()
                            pourcentage_moyen = total_quote_part / total_ht * 100 if total_ht > 0 else 0
                            
                            st.markdown(f"<p class='charges-info'>Total des charges: {total_ht:.2f} € HT</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='charges-info'>Total quote-part: {total_quote_part:.2f} € HT</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='charges-info'>Pourcentage moyen: {pourcentage_moyen:.2f}%</p>", unsafe_allow_html=True)
                        
                        # Afficher le tableau des charges
                        st.subheader("Détail des charges")
                        st.dataframe(charges_data)
                        
                        # Lien de téléchargement
                        st.subheader("Télécharger les données")
                        file_base_name = uploaded_file.name.split('.')[0]
                        download_link = get_table_download_link(charges_data, f"{file_base_name}_charges", output_format)
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Analyse graphique si possible
                        if 'Désignation' in charges_data.columns and 'Quote-part' in charges_data.columns:
                            st.subheader("Analyse graphique")
                            
                            # Préparation des données pour le graphique
                            chart_data = charges_data.dropna(subset=['Désignation', 'Quote-part'])
                            if len(chart_data) > 0:
                                # Limiter aux 10 charges les plus importantes
                                chart_data = chart_data.sort_values('Quote-part', ascending=False).head(10)
                                
                                # Créer un graphique à barres
                                st.bar_chart(chart_data.set_index('Désignation')['Quote-part'])
                                
                                # Afficher la répartition en camembert
                                fig, ax = plt.subplots(figsize=(10, 5))
                                chart_data['Quote-part'].plot(kind='pie', autopct='%1.1f%%', labels=chart_data['Désignation'], ax=ax)
                                ax.set_title('Répartition des principales charges')
                                ax.set_ylabel('')
                                st.pyplot(fig)
                    else:
                        st.error("Aucune donnée de charges n'a pu être extraite du document. Veuillez vérifier que le PDF contient des tableaux de charges structurés.")
                
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de l'extraction: {str(e)}")

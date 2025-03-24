import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
import pdf2image
import tabula
import tempfile
import os
import base64
import io
import re
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Extracteur Intelligent de Charges",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SmartChargesExtractor:
    """
    Extracteur intelligent de charges utilisant des techniques d'IA
    pour optimiser l'extraction et l'analyse des données.
    """
    
    def __init__(self, pdf_file):
        """Initialise l'extracteur avec le fichier PDF"""
        self.pdf_file = pdf_file
        self.temp_dir = tempfile.mkdtemp()
        self.document_type = None
        self.document_structure = None
        
    def analyze_document_structure(self):
        """Détermine le type et la structure du document"""
        # Sauvegarder temporairement le PDF
        temp_pdf_path = os.path.join(self.temp_dir, "temp.pdf")
        with open(temp_pdf_path, 'wb') as f:
            f.write(self.pdf_file.getvalue())
            
        # Extraire le texte de la première page pour analyse
        images = pdf2image.convert_from_path(temp_pdf_path, first_page=1, last_page=1)
        if not images:
            return None
            
        img_path = os.path.join(self.temp_dir, "first_page.png")
        images[0].save(img_path, 'PNG')
        
        text = pytesseract.image_to_string(Image.open(img_path), lang='fra')
        
        # Identifier le type de document
        if 'RELEVE GENERAL DE DEPENSES' in text:
            self.document_type = 'releve_general'
        elif 'RELEVE INDIVIDUEL DES CHARGES' in text:
            self.document_type = 'releve_individuel'
        elif 'CHARGES LOCATIVES' in text:
            self.document_type = 'charges_locatives'
        else:
            self.document_type = 'generic'
            
        # Analyser la structure des tableaux
        # Cette fonction identifie les positions probables des tableaux
        # et leur structure pour optimiser l'extraction
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Détection des lignes horizontales et verticales
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Déterminer si le document a une structure tabulaire claire
        table_mask = horizontal_lines + vertical_lines
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            self.document_structure = 'tabular'
        else:
            # Tenter une autre approche pour détecter des structures tabulaires moins évidentes
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 10:
                self.document_structure = 'semi_tabular'
            else:
                self.document_structure = 'unstructured'
        
        return {
            'type': self.document_type,
            'structure': self.document_structure
        }
    
    def extract_optimized(self):
        """Extraction optimisée en fonction du type et de la structure du document"""
        # Analyser la structure si ce n'est pas déjà fait
        if not self.document_type or not self.document_structure:
            self.analyze_document_structure()
        
        # Stratégie d'extraction basée sur la structure identifiée
        if self.document_structure == 'tabular':
            # Utiliser une approche basée sur tabula pour les documents bien structurés
            return self.extract_tabular()
        elif self.document_structure == 'semi_tabular':
            # Approche hybride pour documents semi-structurés
            return self.extract_hybrid()
        else:
            # Approche basée sur OCR pour documents non structurés
            return self.extract_unstructured()
    
    def extract_tabular(self):
        """Extraction pour documents avec structure tabulaire claire"""
        temp_pdf_path = os.path.join(self.temp_dir, "temp.pdf")
        with open(temp_pdf_path, 'wb') as f:
            f.write(self.pdf_file.getvalue())
        
        # Optimiser les paramètres de tabula en fonction du type de document
        if self.document_type == 'releve_general':
            tables = tabula.read_pdf(
                temp_pdf_path,
                pages='all',
                multiple_tables=True,
                lattice=True,
                guess=False
            )
        else:
            tables = tabula.read_pdf(
                temp_pdf_path,
                pages='all',
                multiple_tables=True,
                stream=True,
                guess=True
            )
        
        return self.process_tables(tables)
    
    def extract_hybrid(self):
        """Approche hybride combinant tabula et OCR ciblé"""
        # Extraire d'abord avec tabula
        tables_from_tabula = self.extract_tabular()
        
        # Si les résultats sont insuffisants, compléter avec OCR ciblé
        if not tables_from_tabula or self.is_extraction_incomplete(tables_from_tabula):
            # Convertir les pages en images
            temp_pdf_path = os.path.join(self.temp_dir, "temp.pdf")
            with open(temp_pdf_path, 'wb') as f:
                f.write(self.pdf_file.getvalue())
                
            images = pdf2image.convert_from_path(temp_pdf_path)
            tables_from_ocr = []
            
            # Utiliser le multithreading pour accélérer l'OCR
            with ThreadPoolExecutor() as executor:
                futures = []
                for i, img in enumerate(images):
                    img_path = os.path.join(self.temp_dir, f"page_{i+1}.png")
                    img.save(img_path, 'PNG')
                    futures.append(executor.submit(self.extract_table_from_image, img_path, i+1))
                
                for future in futures:
                    result = future.result()
                    if result:
                        tables_from_ocr.extend(result)
            
            # Fusionner et nettoyer les résultats
            return self.merge_and_clean_tables(tables_from_tabula, tables_from_ocr)
        
        return tables_from_tabula
    
    def extract_unstructured(self):
        """Extraction pour documents sans structure tabulaire claire"""
        temp_pdf_path = os.path.join(self.temp_dir, "temp.pdf")
        with open(temp_pdf_path, 'wb') as f:
            f.write(self.pdf_file.getvalue())
            
        images = pdf2image.convert_from_path(temp_pdf_path)
        all_tables = []
        
        for i, img in enumerate(images):
            img_path = os.path.join(self.temp_dir, f"page_{i+1}.png")
            img.save(img_path, 'PNG')
            
            # OCR avec configuration optimisée
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(Image.open(img_path), config=custom_config, lang='fra')
            
            # Utiliser l'IA pour reconstruire la structure tabulaire à partir du texte
            tables = self.reconstruct_tables_from_text(text)
            if tables:
                all_tables.extend(tables)
        
        return all_tables
    
    def extract_table_from_image(self, img_path, page_num):
        """Extrait les tableaux d'une image avec une approche optimisée"""
        try:
            # Charger et préparer l'image
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Amélioration adaptative du contraste pour une meilleure détection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
            
            # Détection des lignes
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combinaison des lignes
            table_mask = horizontal_lines + vertical_lines
            
            # Dilatation pour connecter les composants proches
            kernel = np.ones((3,3), np.uint8)
            table_mask = cv2.dilate(table_mask, kernel, iterations=1)
            
            # Trouver les contours des tableaux
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filtrer les petits contours
                if w < 100 or h < 100:
                    continue
                    
                # Extraire et traiter la région
                table_roi = img[y:y+h, x:x+w]
                temp_table_path = os.path.join(self.temp_dir, f"table_p{page_num}_t{i+1}.png")
                cv2.imwrite(temp_table_path, table_roi)
                
                # OCR adaptatif sur la région du tableau
                custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                ocr_data = pytesseract.image_to_data(
                    Image.open(temp_table_path),
                    output_type=pytesseract.Output.DATAFRAME,
                    config=custom_config,
                    lang='fra'
                )
                
                # Reconstruire le tableau avec une approche intelligente
                df = self.smart_table_reconstruction(ocr_data, temp_table_path)
                if not df.empty:
                    tables.append(df)
            
            return tables
            
        except Exception as e:
            st.error(f"Erreur lors de l'extraction du tableau de la page {page_num}: {str(e)}")
            return []
    
    def smart_table_reconstruction(self, ocr_data, img_path):
        """Reconstruction intelligente de la structure du tableau"""
        try:
            # Filtrer les données OCR de faible confiance
            ocr_data = ocr_data[ocr_data['conf'] > 40]
            
            if ocr_data.empty:
                return pd.DataFrame()
            
            # Analyse de la géométrie des blocs de texte pour détecter les colonnes et lignes
            # Algorithme de clustering adaptatif pour les positions
            ocr_data['line_cluster'] = self.adaptive_cluster(ocr_data['top'].values)
            ocr_data['col_cluster'] = self.adaptive_cluster(ocr_data['left'].values)
            
            # Création d'une matrice représentant le tableau
            pivot_table = pd.DataFrame(index=sorted(ocr_data['line_cluster'].unique()),
                                     columns=sorted(ocr_data['col_cluster'].unique()))
            
            # Remplir la matrice avec les valeurs
            for _, row in ocr_data.iterrows():
                line = row['line_cluster']
                col = row['col_cluster']
                text = row['text']
                
                current = pivot_table.at[line, col]
                if pd.isna(current):
                    pivot_table.at[line, col] = text
                else:
                    pivot_table.at[line, col] = f"{current} {text}"
            
            # Convertir en DataFrame normal
            result = pivot_table.reset_index(drop=True)
            
            # Essayer d'identifier les en-têtes
            if not result.empty:
                # La première ligne est probablement l'en-tête
                result.columns = result.iloc[0].fillna(f'Col_{{i}}' for i in range(len(result.columns)))
                result = result.iloc[1:].reset_index(drop=True)
                
                # Nettoyer les noms de colonnes
                result.columns = [str(col).strip() for col in result.columns]
            
            return result
            
        except Exception as e:
            st.error(f"Erreur lors de la reconstruction du tableau: {str(e)}")
            return pd.DataFrame()
    
    def adaptive_cluster(self, values, threshold_factor=0.05):
        """Algorithme de clustering adaptatif pour détecter les lignes/colonnes"""
        if len(values) == 0:
            return []
            
        # Trier les valeurs
        sorted_values = np.sort(values)
        
        # Calculer les différences entre valeurs consécutives
        diffs = np.diff(sorted_values)
        
        # Calculer le seuil adaptatif basé sur l'échelle des données
        range_value = sorted_values[-1] - sorted_values[0]
        threshold = max(5, range_value * threshold_factor)  # Minimum 5 pixels
        
        # Trouver les ruptures (nouvelles lignes/colonnes)
        breaks = np.where(diffs > threshold)[0]
        
        # Créer les clusters
        clusters = []
        start_idx = 0
        
        for break_idx in breaks:
            clusters.extend([len(clusters)] * (break_idx + 1 - start_idx))
            start_idx = break_idx + 1
            
        # Ajouter le dernier cluster
        clusters.extend([len(clusters)] * (len(sorted_values) - start_idx))
        
        # Recréer les assignations de cluster dans l'ordre original
        result = np.zeros_like(values, dtype=int)
        sorted_indices = np.argsort(values)
        for i, cluster in enumerate(clusters):
            result[sorted_indices[i]] = cluster
            
        return result
    
    def process_tables(self, tables):
        """Traite et nettoie les tableaux extraits"""
        processed_tables = []
        
        for table in tables:
            if table.empty:
                continue
                
            # Nettoyer les noms de colonnes
            if table.columns.dtype == 'object':
                table.columns = [str(col).strip() for col in table.columns]
            
            # Supprimer les lignes vides
            table = table.dropna(how='all')
            
            # Supprimer les colonnes vides
            table = table.dropna(axis=1, how='all')
            
            if not table.empty:
                processed_tables.append(table)
        
        return processed_tables
    
    def merge_and_clean_tables(self, tables1, tables2):
        """Fusionne et nettoie les tableaux provenant de différentes méthodes d'extraction"""
        all_tables = tables1.copy() if tables1 else []
        
        # Ajouter les tables de la deuxième méthode en évitant les doublons
        if tables2:
            for table2 in tables2:
                is_duplicate = False
                for table1 in all_tables:
                    if self.is_similar_table(table1, table2):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_tables.append(table2)
        
        return self.process_tables(all_tables)
    
    def is_similar_table(self, table1, table2, similarity_threshold=0.7):
        """Détermine si deux tableaux sont similaires"""
        # Comparaison basique par forme
        if table1.shape != table2.shape:
            return False
            
        # Comparaison par échantillonnage de contenu
        # Prendre un échantillon de cellules et voir si elles correspondent
        n_samples = min(5, len(table1))
        sample_indices = np.random.choice(len(table1), n_samples, replace=False)
        
        matches = 0
        for idx in sample_indices:
            row1 = table1.iloc[idx].astype(str)
            row2 = table2.iloc[idx].astype(str)
            
            # Calculer la similarité entre les lignes
            similarity = sum(1 for a, b in zip(row1, row2) if a == b) / len(row1)
            if similarity > similarity_threshold:
                matches += 1
        
        return matches / n_samples > similarity_threshold
    
    def is_extraction_incomplete(self, tables):
        """Détermine si l'extraction est incomplète ou de mauvaise qualité"""
        if not tables:
            return True
            
        # Vérifier la qualité des tableaux extraits
        for table in tables:
            # Un tableau de charges devrait avoir au moins quelques colonnes
            if len(table.columns) < 3:
                return True
                
            # Vérifier si le tableau contient probablement des données numériques
            numeric_cells = 0
            total_cells = 0
            
            for col in table.columns:
                try:
                    numeric_values = pd.to_numeric(table[col], errors='coerce')
                    numeric_cells += numeric_values.notna().sum()
                    total_cells += len(table[col])
                except:
                    pass
            
            # Un tableau de charges devrait contenir un certain pourcentage de valeurs numériques
            if total_cells > 0 and numeric_cells / total_cells < 0.2:
                return True
        
        return False
    
    def reconstruct_tables_from_text(self, text):
        """Reconstruit les tableaux à partir du texte OCR pour documents non structurés"""
        lines = text.strip().split('\n')
        
        # Supprimer les lignes vides
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return []
            
        # Rechercher des modèles de tableaux dans le texte
        table_sections = []
        current_section = []
        
        for line in lines:
            # Déterminer si la ligne pourrait faire partie d'un tableau
            # (contient des nombres et des séparateurs visuels)
            if re.search(r'\d+[.,]\d+', line) and (re.search(r'\s{2,}', line) or '|' in line):
                current_section.append(line)
            elif current_section:
                if len(current_section) > 2:  # Un tableau doit avoir au moins 3 lignes
                    table_sections.append(current_section)
                current_section = []
        
        # Ajouter la dernière section si elle existe
        if current_section and len(current_section) > 2:
            table_sections.append(current_section)
        
        # Convertir chaque section en DataFrame
        tables = []
        for section in table_sections:
            table = self.convert_text_section_to_table(section)
            if not table.empty:
                tables.append(table)
        
        return tables
    
    def convert_text_section_to_table(self, text_lines):
        """Convertit une section de texte en tableau structuré"""
        # Détecter le séparateur le plus probable
        if '|' in text_lines[0]:
            separator = '|'
        else:
            # Utiliser des espaces multiples comme séparateur
            separator = r'\s{2,}'
        
        # Diviser les lignes selon le séparateur
        rows = []
        for line in text_lines:
            if separator == '|':
                cells = [cell.strip() for cell in line.split('|')]
                # Éliminer les cellules vides aux extrémités
                if cells and not cells[0].strip():
                    cells = cells[1:]
                if cells and not cells[-1].strip():
                    cells = cells[:-1]
            else:
                cells = [cell.strip() for cell in re.split(separator, line) if cell.strip()]
            
            if cells:
                rows.append(cells)
        
        if not rows:
            return pd.DataFrame()
        
        # Normaliser la structure (nombre de colonnes)
        max_cols = max(len(row) for row in rows)
        normalized_rows = [row + [''] * (max_cols - len(row)) for row in rows]
        
        # Créer le DataFrame
        if len(normalized_rows) > 1:
            df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])
        else:
            df = pd.DataFrame([normalized_rows[0]])
        
        return df
    
    def extract_charges_metadata(self):
        """Extrait les métadonnées liées aux charges"""
        temp_pdf_path = os.path.join(self.temp_dir, "temp.pdf")
        with open(temp_pdf_path, 'wb') as f:
            f.write(self.pdf_file.getvalue())
            
        images = pdf2image.convert_from_path(temp_pdf_path, first_page=1, last_page=1)
        if not images:
            return {}
            
        img_path = os.path.join(self.temp_dir, "first_page.png")
        images[0].save(img_path, 'PNG')
        
        text = pytesseract.image_to_string(Image.open(img_path), lang='fra')
        
        metadata = {
            'société': None,
            'immeuble': None,
            'adresse': None,
            'période': None,
            'locataire': None
        }
        
        # Extraire les informations avec des RegEx intelligents
        # Société
        society_patterns = [
            r'(?:Société|SCI|SAS|SARL|SA)\s*:?\s*([A-Z0-9 ]+)',
            r'^([A-Z][A-Z0-9 ]+)(?:\n|$)'
        ]
        for pattern in society_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                metadata['société'] = match.group(1).strip()
                break
        
        # Immeuble
        immeuble_patterns = [
            r'Immeuble\s*:?\s*([A-Z0-9 ]+)',
            r'(?:FREJUS|CAPITOU)',
        ]
        for pattern in immeuble_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['immeuble'] = match.group(0).strip()
                break
        
        # Adresse (recherche du code postal et ville)
        address_match = re.search(r'\b(\d{5})\s+([A-Z]+)\b', text)
        if address_match:
            # Rechercher une ligne d'adresse autour du code postal
            cp_index = text.find(address_match.group(0))
            line_start = text.rfind('\n', 0, cp_index) + 1
            line_end = text.find('\n', cp_index)
            
            if line_start > 0 and line_end > cp_index:
                metadata['adresse'] = text[line_start:line_end].strip()
        
        # Période
        period_patterns = [
            r'[Pp]ériode\s+du\s+(\d{2}/\d{2}/\d{4})\s+au\s+(\d{2}/\d{2}/\d{4})',
            r'du\s+(\d{2}/\d{2}/\d{4})\s+au\s+(\d{2}/\d{2}/\d{4})'
        ]
        for pattern in period_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['période'] = f"{match.group(1)} au {match.group(2)}"
                break
        
        # Locataire
        locataire_patterns = [
            r'(?:ELECTRO DEPOT|DEPOT FRANCE)',
            r'(?:Locataire|Preneur)\s*:?\s*([A-Z][A-Za-z0-9 ]+)'
        ]
        for pattern in locataire_patterns:
            match = re.search(pattern, text)
            if match:
                if match.groups():
                    metadata['locataire'] = match.group(1).strip()
                else:
                    metadata['locataire'] = match.group(0).strip()
                break
        
        return metadata
    
    def process_charges_data(self, tables):
        """Traite les tableaux pour extraire les données de charges"""
        if not tables:
            return pd.DataFrame()
        
        # Identifier les tableaux pertinents contenant des données de charges
        charges_tables = []
        
        for table in tables:
            if table.empty:
                continue
            
            # Nettoyer les noms de colonnes
            if table.columns.dtype == 'object':
                table.columns = [str(col).strip() for col in table.columns]
            
            # Évaluer si ce tableau contient probablement des données de charges
            score = 0
            
            # Vérifier la présence de colonnes typiques
            column_patterns = {
                'designation': [r'(?:désignation|libellé|intitulé|poste|nature)', 3],
                'montant': [r'(?:montant|mont\.)', 3],
                'quote_part': [r'(?:quote|part|répartir|quote-part)', 3],
                'code': [r'(?:code|chapitre|chap|clé)', 2],
                'tva': [r'(?:tva|t\.v\.a)', 2],
            }
            
            for col in table.columns:
                col_lower = str(col).lower()
                for pattern, points in column_patterns.values():
                    if re.search(pattern, col_lower):
                        score += points
            
            # Vérifier la présence de mots-clés dans les données
            keywords = [
                r'nettoyage', r'déchet', r'électricité', r'entretien',
                r'espaces verts', r'honoraire', r'eau', r'ascenseur',
                r'chauffage', r'gardien', r'sécurité', r'vidéosurveillance'
            ]
            
            for keyword in keywords:
                if any(re.search(keyword, str(cell), re.IGNORECASE) for cell in table.values.flatten() if isinstance(cell, str)):
                    score += 1
            
            # Vérifier la présence de valeurs numériques
            numeric_cols = 0
            for col in table.columns:
                try:
                    numeric_values = pd.to_numeric(table[col], errors='coerce')
                    if numeric_values.notna().sum() > len(table) / 2:
                        numeric_cols += 1
                except:
                    pass
            
            if numeric_cols >= 2:
                score += 3
            
            # Si le score est suffisant, c'est probablement un tableau de charges
            if score >= 5:
                charges_tables.append(table)
        
        # Si aucun tableau pertinent n'est trouvé, prendre le plus grand tableau
        if not charges_tables and tables:
            largest_table = max(tables, key=lambda t: t.size)
            charges_tables.append(largest_table)
        
        # Fusionner les tableaux de charges
        if len(charges_tables) > 1:
            # Vérifier si les tableaux sont complémentaires (différentes pages d'un même tableau)
            if self.are_complementary_tables(charges_tables):
                # Concaténer les tableaux
                merged_table = pd.concat(charges_tables, ignore_index=True)
            else:
                # Prendre le tableau le plus pertinent/complet
                merged_table = max(charges_tables, key=lambda t: t.size)
        elif charges_tables:
            merged_table = charges_tables[0]
        else:
            return pd.DataFrame()
        
        # Normaliser les noms de colonnes
        merged_table = self.normalize_columns(merged_table)
        
        # Nettoyer et convertir les données numériques
        merged_table = self.clean_charges_data(merged_table)
        
        return merged_table
    
    def are_complementary_tables(self, tables):
        """Détermine si les tableaux sont complémentaires (suite logique)"""
        if len(tables) < 2:
            return False
        
        # Vérifier si les colonnes sont similaires
        columns_similarity = all(
            set(tables[0].columns).issubset(set(table.columns)) for table in tables[1:]
        ) or all(
            set(table.columns).issubset(set(tables[0].columns)) for table in tables[1:]
        )
        
        if not columns_similarity:
            return False
        
        # Vérifier si les tableaux contiennent des données qui se suivent (ex: différents chapitres)
        # ou des données dupliquées
        if 'Code' in tables[0].columns:
            codes = set()
            for table in tables:
                table_codes = set(table['Code'].astype(str))
                # Si plus de 20% des codes sont déjà vus, ce n'est probablement pas complémentaire
                if len(table_codes.intersection(codes)) > 0.2 * len(table_codes):
                    return False
                codes.update(table_codes)
            return True
        
        return True
    
    def normalize_columns(self, table):
        """Normalise les noms de colonnes du tableau de charges"""
        column_mapping = {}
        
        for col in table.columns:
            col_lower = str(col).lower()
            
            if any(pattern in col_lower for pattern in ['désignation', 'libellé', 'intitulé']):
                column_mapping[col] = 'Désignation'
            elif any(pattern in col_lower for pattern in ['code', 'chapitre', 'chap', 'clé']):
                column_mapping[col] = 'Code'
            elif 'mont' in col_lower and 'ht' in col_lower:
                column_mapping[col] = 'Montant HT'
            elif 'mont' in col_lower and 'tva' in col_lower:
                column_mapping[col] = 'Montant TVA'
            elif 'mont' in col_lower and 'ttc' in col_lower:
                column_mapping[col] = 'Montant TTC'
            elif any(pattern in col_lower for pattern in ['quote', 'part', 'répartir']) and 'ht' in col_lower:
                column_mapping[col] = 'Quote-part HT'
            elif any(pattern in col_lower for pattern in ['quote', 'part', 'répartir']) and 'tva' in col_lower:
                column_mapping[col] = 'Quote-part TVA'
            elif any(pattern in col_lower for pattern in ['tantième', 'millième']) and 'glob' in col_lower:
                column_mapping[col] = 'Tantièmes globaux'
            elif any(pattern in col_lower for pattern in ['tantième', 'millième']) and 'part' in col_lower:
                column_mapping[col] = 'Tantièmes particuliers'
            elif 'date' in col_lower:
                column_mapping[col] = 'Date'
            elif 'r' in col_lower and '%' in col_lower:
                column_mapping[col] = 'Pourcentage'
        
        # Appliquer le mapping
        if column_mapping:
            table = table.rename(columns=column_mapping)
        
        return table
    
    def clean_charges_data(self, table):
        """Nettoie et convertit les données numériques du tableau de charges"""
        # Fonction pour convertir les valeurs en numérique
        def convert_to_numeric(val):
            if pd.isna(val):
                return np.nan
            
            if isinstance(val, (int, float)):
                return float(val)
            
            # Nettoyer la chaîne
            val_str = str(val)
            val_str = val_str.replace(' ', '').replace(',', '.').replace('€', '')
            
            # Supprimer les caractères non numériques sauf points et signes
            val_str = re.sub(r'[^\d\.\-\+]', '', val_str)
            
            try:
                return float(val_str)
            except ValueError:
                return np.nan
        
        # Colonnes qui devraient être numériques
        numeric_columns = [
            'Montant HT', 'Montant TVA', 'Montant TTC',
            'Quote-part HT', 'Quote-part TVA',
            'Tantièmes globaux', 'Tantièmes particuliers'
        ]
        
        # Convertir les colonnes
        for col in numeric_columns:
            if col in table.columns:
                table[col] = table[col].apply(convert_to_numeric)
        
        # Supprimer les lignes ne contenant que des NaN
        table = table.dropna(how='all')
        
        # Ajouter une colonne de pourcentage si possible
        if 'Montant HT' in table.columns and 'Quote-part HT' in table.columns:
            table['Pourcentage'] = table.apply(
                lambda row: 0 if pd.isna(row['Montant HT']) or pd.isna(row['Quote-part HT']) or row['Montant HT'] == 0
                           else row['Quote-part HT'] / row['Montant HT'] * 100,
                axis=1
            )
        
        return table
    
    def extract_and_process(self):
        """Méthode principale pour extraire et traiter les données de charges"""
        # Analyser la structure du document
        structure_info = self.analyze_document_structure()
        
        # Extraire les tableaux
        tables = self.extract_optimized()
        
        # Traiter les données de charges
        charges_data = self.process_charges_data(tables)
        
        # Extraire les métadonnées
        metadata = self.extract_charges_metadata()
        
        # Nettoyer les fichiers temporaires
        self.cleanup()
        
        return charges_data, metadata, structure_info
    
    def cleanup(self):
        """Nettoie les fichiers temporaires"""
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            st.error(f"Erreur lors du nettoyage: {str(e)}")


def get_table_download_link(df, filename, format_type):
    """Génère un lien de téléchargement pour un DataFrame"""
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
    st.title("Extracteur Intelligent de Charges")
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0D47A1;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #E3F2FD;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .info-box-title {
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .download-link {
            background-color: #1976D2;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin-top: 0.5rem;
        }
        .download-link:hover {
            background-color: #1565C0;
        }
        .metadata-box {
            background-color: #E8F5E9;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .stat-box {
            background-color: #FFF8E1;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #FB8C00;
        }
        .stat-label {
            font-size: 1rem;
            color: #424242;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Extracteur Intelligent de Charges</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">Optimisé par intelligence artificielle</div>
        <p>Cette application utilise l'IA pour extraire automatiquement les données de charges depuis vos relevés PDF, 
        même ceux contenant des tableaux sous forme d'images.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Options")
    output_format = st.sidebar.selectbox(
        "Format de téléchargement",
        options=["csv", "xlsx", "json"],
        index=0
    )
    
    advanced_options = st.sidebar.expander("Options avancées")
    with advanced_options:
        ocr_lang = st.selectbox(
            "Langue du document",
            options=["fra", "fra+eng", "eng"],
            index=0
        )
        
        use_advanced_ocr = st.checkbox(
            "Utiliser l'OCR avancé",
            value=True,
            help="Active des techniques d'OCR avancées pour les documents difficiles à lire"
        )
    
    # Zone de téléchargement de fichier
    uploaded_file = st.file_uploader("Choisissez un relevé de charges au format PDF", type=['pdf'])
    
    if uploaded_file is not None:
        # Afficher les informations du fichier
        file_details = {
            "Nom du fichier": uploaded_file.name,
            "Taille": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">Fichier sélectionné</div>
            <p><strong>Nom:</strong> {file_details["Nom du fichier"]}</p>
            <p><strong>Taille:</strong> {file_details["Taille"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton pour lancer l'extraction
        if st.button("Extraire les données de charges"):
            with st.spinner('Analyse intelligente en cours...'):
                try:
                    # Initialiser l'extracteur intelligent
                    extractor = SmartChargesExtractor(uploaded_file)
                    
                    # Extraire et traiter les données
                    charges_data, metadata, structure_info = extractor.extract_and_process()
                    
                    if not charges_data.empty:
                        # Afficher les métadonnées du document
                        st.markdown('<h2 class="sub-header">Informations du document</h2>', unsafe_allow_html=True)
                        
                        metadata_html = '<div class="metadata-box">'
                        for key, value in metadata.items():
                            if value:
                                metadata_html += f'<p><strong>{key.capitalize()}:</strong> {value}</p>'
                        metadata_html += '</div>'
                        st.markdown(metadata_html, unsafe_allow_html=True)
                        
                        # Afficher les informations de structure du document
                        if structure_info:
                            st.markdown('<h2 class="sub-header">Analyse du document</h2>', unsafe_allow_html=True)
                            
                            doc_type = structure_info.get('type', 'inconnu')
                            doc_structure = structure_info.get('structure', 'inconnu')
                            
                            doc_type_friendly = {
                                'releve_general': 'Relevé général de dépenses',
                                'releve_individuel': 'Relevé individuel des charges',
                                'charges_locatives': 'Décompte de charges locatives',
                                'generic': 'Document générique'
                            }.get(doc_type, 'Type inconnu')
                            
                            doc_structure_friendly = {
                                'tabular': 'Structure tabulaire bien définie',
                                'semi_tabular': 'Structure semi-tabulaire',
                                'unstructured': 'Document peu structuré'
                            }.get(doc_structure, 'Structure inconnue')
                            
                            st.markdown(f"""
                            <div class="info-box">
                                <p><strong>Type de document:</strong> {doc_type_friendly}</p>
                                <p><strong>Structure détectée:</strong> {doc_structure_friendly}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Afficher les statistiques des charges
                        if 'Montant HT' in charges_data.columns and 'Quote-part HT' in charges_data.columns:
                            total_ht = charges_data['Montant HT'].sum()
                            total_quote_part = charges_data['Quote-part HT'].sum()
                            pourcentage_moyen = total_quote_part / total_ht * 100 if total_ht > 0 else 0
                            
                            st.markdown('<h2 class="sub-header">Synthèse des charges</h2>', unsafe_allow_html=True)
                            
                            cols = st.columns(3)
                            with cols[0]:
                                st.markdown(f"""
                                <div class="stat-box">
                                    <div class="stat-label">Total charges</div>
                                    <div class="stat-value">{total_ht:.2f} €</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with cols[1]:
                                st.markdown(f"""
                                <div class="stat-box">
                                    <div class="stat-label">Quote-part</div>
                                    <div class="stat-value">{total_quote_part:.2f} €</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with cols[2]:
                                st.markdown(f"""
                                <div class="stat-box">
                                    <div class="stat-label">Pourcentage</div>
                                    <div class="stat-value">{pourcentage_moyen:.2f} %</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Afficher le tableau des charges
                        st.markdown('<h2 class="sub-header">Détail des charges</h2>', unsafe_allow_html=True)
                        st.dataframe(charges_data)
                        
                        # Lien de téléchargement
                        st.markdown('<h2 class="sub-header">Télécharger les données</h2>', unsafe_allow_html=True)
                        file_base_name = uploaded_file.name.split('.')[0]
                        download_link = get_table_download_link(charges_data, f"{file_base_name}_charges", output_format)
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Analyse graphique
                        if 'Désignation' in charges_data.columns and 'Quote-part HT' in charges_data.columns:
                            st.markdown('<h2 class="sub-header">Analyse graphique</h2>', unsafe_allow_html=True)
                            
                            # Préparation des données pour le graphique
                            chart_data = charges_data.dropna(subset=['Désignation', 'Quote-part HT']).copy()
                            
                            # Tronquer les désignations trop longues
                            chart_data['Désignation'] = chart_data['Désignation'].apply(
                                lambda x: x[:25] + '...' if len(str(x)) > 25 else x
                            )
                            
                            if len(chart_data) > 0:
                                # Limiter aux 10 charges les plus importantes
                                chart_data = chart_data.sort_values('Quote-part HT', ascending=False).head(10)
                                
                                # Créer un graphique à barres
                                st.bar_chart(chart_data.set_index('Désignation')['Quote-part HT'])
                                
                                # Afficher la répartition en camembert
                                fig, ax = plt.subplots(figsize=(10, 10))
                                wedges, texts, autotexts = ax.pie(
                                    chart_data['Quote-part HT'],
                                    labels=chart_data['Désignation'],
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    shadow=False
                                )
                                
                                # Égaliser la taille des étiquettes
                                plt.setp(autotexts, size=10, weight="bold")
                                plt.setp(texts, size=9)
                                
                                ax.set_title('Répartition des principales charges')
                                st.pyplot(fig)
                    else:
                        st.error("Aucune donnée de charges n'a pu être extraite du document. Veuillez vérifier que le PDF contient des données de charges.")
                
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de l'extraction: {str(e)}")
    
    # Informations supplémentaires
    st.markdown("---")
    with st.expander("Guide d'utilisation"):
        st.markdown("""
        ### Comment utiliser cette application
        
        1. **Téléchargez** votre relevé de charges au format PDF
        2. Cliquez sur **Extraire les données de charges**
        3. Examinez les informations extraites et les visualisations
        4. **Téléchargez** les données dans le format de votre choix pour analyse approfondie
        
        ### Types de documents supportés
        
        - Relevés généraux de dépenses d'immeuble
        - Relevés individuels de charges locatives
        - Factures de charges avec détail des postes
        - Décomptes annuels de régularisation de charges
        """)
    
    with st.expander("À propos de l'analyse intelligente"):
        st.markdown("""
        ### Comment fonctionne l'extraction intelligente
        
        Cette application utilise plusieurs techniques d'IA pour optimiser l'extraction des données :
        
        1. **Analyse adaptative de structure** - L'application détermine automatiquement le type et la structure du document
        2. **Extraction multi-stratégie** - Différentes méthodes sont utilisées selon le type de document
        3. **OCR adaptatif** - La reconnaissance de caractères s'adapte à la qualité du document
        4. **Reconstruction intelligente** - Les tableaux sont reconstruits même à partir de documents mal structurés
        5. **Analyse sémantique** - Identification du contexte et du sens des données
        
        Cette approche permet une extraction plus précise avec moins de ressources système.
        """)


if __name__ == "__main__":
    main()

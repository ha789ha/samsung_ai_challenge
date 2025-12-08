import pandas as pd
import numpy as np
from utils import bbox_iou

def clean_predictions(df):
    if df.empty:
        return df

    df["bbox_tuple"] = df["bbox"].apply(lambda x: tuple(map(int, x.split(","))))
    df = df.sort_values('confidence_score', ascending=False)
    df = df.drop_duplicates(subset=['ID', 'bbox_tuple'], keep='first')
    df = df[~df["category_type"].isin(["abandon", "table_caption", "table_footnote", "formula_caption", "figure_caption"])].copy()
    
    cleaned_df_list = []
    for img_id, group in df.groupby("ID"):
        group_copy = group.copy()
        
        title_rows = group_copy[group_copy["category_type"] == "title"]
        if len(title_rows) > 1:
            true_title_idx = title_rows['bbox_tuple'].apply(lambda bbox: bbox[1]).idxmin()
            other_header_indices = title_rows.index.drop(true_title_idx)
            group_copy.loc[other_header_indices, 'category_type'] = 'subtitle'
            group_copy.loc[true_title_idx, 'category_type'] = 'title'

        indices_to_remove = set()
        rows_as_tuples = list(group_copy.itertuples())

        for outer_row in rows_as_tuples:
            if outer_row.Index in indices_to_remove:
                continue

            contained_count = 0
            for inner_row in rows_as_tuples:
                if outer_row.Index == inner_row.Index:
                    continue
                
                outer_bbox = outer_row.bbox_tuple
                inner_bbox = inner_row.bbox_tuple
                if (outer_bbox[0] <= inner_bbox[0] and outer_bbox[1] <= inner_bbox[1] and
                    outer_bbox[2] >= inner_bbox[2] and outer_bbox[3] >= inner_bbox[3]):
                    contained_count += 1
            
            if contained_count >= 2:
                indices_to_remove.add(outer_row.Index)
        
        group_copy.drop(list(indices_to_remove), inplace=True, errors='ignore')

        group_copy = group_copy.sort_values(by="bbox_tuple", key=lambda b: b.apply(lambda x: (x[1], x[0])))

        keep = []
        for _, row in group_copy.iterrows():
            is_overlapping = False
            for kept_row in keep:
                if bbox_iou(row["bbox_tuple"], kept_row["bbox_tuple"]) > 0.3:
                    is_overlapping = True
                    break
            if not is_overlapping:
                keep.append(row.to_dict())

        for order, element in enumerate(keep):
            element["order"] = order
        
        if keep:
            cleaned_df_list.append(pd.DataFrame(keep))

    if not cleaned_df_list:
        return pd.DataFrame()

    df_clean = pd.concat(cleaned_df_list, ignore_index=True)
    if "bbox_tuple" in df_clean.columns:
        df_clean.drop(columns=["bbox_tuple"], inplace=True)
        
    return df_clean

def process_and_order_document(df):
    # bbox 열을 tuple로 변환 및 좌표 컬럼 추가
    df['bbox_tuple'] = df['bbox'].apply(lambda x: tuple(map(int, x.split(','))))
    df['y_top'] = df['bbox_tuple'].apply(lambda b: b[1])
    df['y_bottom'] = df['bbox_tuple'].apply(lambda b: b[3])
    df['x_left'] = df['bbox_tuple'].apply(lambda b: b[0])
    df['x_right'] = df['bbox_tuple'].apply(lambda b: b[2])

    def process_document(doc_df):
        headers = doc_df[doc_df['category_type'].isin(['title', 'subtitle'])].sort_values(by='y_top').copy()
        contents = doc_df[~doc_df['category_type'].isin(['title', 'subtitle'])].copy()
        
        if headers.empty:
            return contents.sort_values(by=['y_top', 'x_left']).to_dict('records')

        header_groups = []
        current_group = []
        threshold = 20

        for _, header in headers.iterrows():
            header_dict = header.to_dict()
            if header_dict['category_type'] == 'title':
                if current_group:
                    header_groups.append(sorted(current_group, key=lambda h: h['x_left']))
                current_group = []
                header_groups.append([header_dict])
            else:
                if not current_group or abs(header_dict['y_top'] - current_group[-1]['y_top']) <= threshold:
                    current_group.append(header_dict)
                else:
                    header_groups.append(sorted(current_group, key=lambda h: h['x_left']))
                    current_group = [header_dict]
        if current_group:
            header_groups.append(sorted(current_group, key=lambda h: h['x_left']))

        final_ordered_elements = []
        remaining_contents = contents.copy()

        first_header_y = header_groups[0][0]['y_top']
        content_before = remaining_contents[remaining_contents['y_top'] < first_header_y]
        if not content_before.empty:
            sorted_content = content_before.sort_values(by=['y_top', 'x_left'])
            final_ordered_elements.extend(sorted_content.to_dict('records'))
            remaining_contents.drop(sorted_content.index, inplace=True)

        for i, group in enumerate(header_groups):
            group_y_bottom = max(h['y_bottom'] for h in group)
            next_group_y_top = float('inf')
            if i + 1 < len(header_groups):
                next_group_y_top = header_groups[i+1][0]['y_top']
            
            group_contents_df = remaining_contents[
                (remaining_contents['y_top'] >= group_y_bottom) &
                (remaining_contents['y_top'] < next_group_y_top)
            ]

            if len(group) > 1:
                sub_boundaries = [sub['x_left'] for sub in group[1:]] + [float('inf')]
                conditions = [group_contents_df['x_right'] <= b for b in sub_boundaries]
                choices = list(range(len(sub_boundaries)))
                group_contents_df['subtitle_index'] = np.select(conditions, choices, default=len(choices)-1)
                
                for sub_idx, subtitle in enumerate(group):
                    final_ordered_elements.append(subtitle)
                    
                    sub_content = group_contents_df[group_contents_df['subtitle_index'] == sub_idx]
                    if not sub_content.empty:
                        sorted_sub_content = sub_content.sort_values(by=['y_top', 'x_left'])
                        final_ordered_elements.extend(sorted_sub_content.to_dict('records'))
            
            else:
                final_ordered_elements.extend(group)
                if not group_contents_df.empty:
                    sorted_content = group_contents_df.sort_values(by=['y_top', 'x_left'])
                    final_ordered_elements.extend(sorted_content.to_dict('records'))
            
            if not group_contents_df.empty:
                remaining_contents.drop(group_contents_df.index, inplace=True)

        return final_ordered_elements

    # ID별로 처리
    all_results = []
    for _, group_df in df.groupby('ID'):
        all_results.extend(process_document(group_df))

    final_df = pd.DataFrame(all_results)
    
    # order 열 추가
    final_df['order'] = final_df.groupby('ID').cumcount()
    
    # bbox 컬럼 복원
    final_df['bbox'] = final_df['bbox_tuple'].apply(lambda x: ','.join(map(str, x)))

    # 최종 컬럼 순서
    final_columns = ['ID', 'category_type', 'confidence_score', 'order', 'text', 'bbox']
    final_df_columns = [col for col in final_columns if col in final_df.columns]
    final_df = final_df[final_df_columns]

    return final_df
import json
import os
import argparse
import re
import random
from collections import Counter
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def reshape_category(CategoryName):
    if CategoryName == "":
        return ""
    category_list = CategoryName.split('--')
    reshaped_category_name = ""
    for i, category in enumerate(category_list):
        category = category.strip()
        if category != "" and category.lower() != "others" and category.lower() != "other" and category.lower() != "unspecified":
            category = ' '.join([word for word in category.split() if word.lower() != "other"])
            if i == 0:
                reshaped_category_name += category
            else:
                reshaped_category_name += " -- " + category
    return reshaped_category_name

def construct_message(user_prompt_template, detail_info, Asset_list, data_idx, AssetCnt, AssetType, FullLanguage):
    user_prompt_template = user_prompt_template.format(AssetCnt, AssetType, FullLanguage)
    user_message = {"content": user_prompt_template + detail_info, "role": "user"}
    assistant_content = ""
    for asset in Asset_list:
        assistant_content += "Ad:" + asset + "\n"
    assistant_message = {"content": assistant_content, "role": "assistant"}
    message = {"prompt_id": str(data_idx), "messages": [user_message, assistant_message]}

    return message

def get_repeated_string_indices(string_list):
    count = Counter(string_list)
    candidates = [string for string, freq in count.items() if freq >= 2 and string != ""]

    if not candidates:
        return None, []

    random_string = random.choice(candidates)
    indices = [i for i, s in enumerate(string_list) if s == random_string]

    return random_string, indices


def main(args):
    inputfile = args.input

    input_row = 0
    data_idx = 0
    data_withDKI = 0
    data_withInsight = 0
    full_data_list = []

    user_prompt_template = "Please generate {} Ad {} in {} language, based on the following information:\n"
    with open(inputfile, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            FinalUrl, Domain, CategoryName, DescriptionOfAdvertiser, FullLanguage, AssetType, JointAsset, JointIsDKI, JointInsight, sd_doc = line.split('\t')
            '''
            print("FinalUrl: ", FinalUrl)
            print("Domain: ", Domain)
            print("CategoryName: ", CategoryName)
            print("DescriptionOfAdvertiser: ", DescriptionOfAdvertiser)
            print("FullLanguage: ", FullLanguage)
            print("AssetType: ", AssetType)
            print("JointAsset: ", JointAsset)
            print("JointIsDKI: ", JointIsDKI)
            print("JointInsight: ", JointInsight)
            print("sd_doc: ", sd_doc)
            '''
            Asset_list = JointAsset.split('[SEP]')
            Asset_list = [asset.strip() for asset in Asset_list]
            IsDKI_list = JointIsDKI.split('[SEP]')
            IsDKI_list = [isDKI.strip() for isDKI in IsDKI_list]
            Insight_list = JointInsight.split('[SEP]')
            Insight_list = [insight.strip() for insight in Insight_list]

            detail_info = "FinalUrl: " + FinalUrl + " \n"
            if len(Domain) > 2:
                detail_info += "Domain: " + Domain + " \n"
            reshaped_category_name = reshape_category(CategoryName)
            if len(reshaped_category_name) > 2:
                detail_info += "Category: " + reshaped_category_name + " \n"
            if len(DescriptionOfAdvertiser) > 2:
                detail_info += "DescriptionOfAdvertiser: " + DescriptionOfAdvertiser + " \n"
            if len(sd_doc) > 2:
                delimiters = ";\n"
                regexPattern = '|'.join(map(re.escape, delimiters))
                text_list = re.split(regexPattern, sd_doc)
                cleaned_text_list = [sent for sent in text_list if len(sent) > 3]
                SD_text = "; ".join(cleaned_text_list)
                sd_doc = SD_text[:400]
                detail_info += "LandingPage: " + sd_doc + " \n"
            if AssetType == "Headline":
                detail_info += "CharacterLimit: between 10 to 20 characters. \n"
            if AssetType == "Description":
                min_length = min([len(asset) for asset in Asset_list])
                detail_info += "CharacterLimit: between " + str(min_length) + " to 90 characters. \n"

            if "1" in IsDKI_list:
                DKI_Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "1"]
                DKI_Asset_list = list(set(DKI_Asset_list))
                AssetCnt = len(DKI_Asset_list)
                Insight = "Incorporate dynamic keyword insertion to make your ad more relevant to query."
                detail_DKI_info = detail_info + "Insight: " + Insight + " \n"

                message = construct_message(user_prompt_template, detail_DKI_info, DKI_Asset_list, data_idx, AssetCnt, AssetType, FullLanguage)
                #print("IsDKI message: ", message)
                full_data_list.append(message)
                data_idx += 1
                data_withDKI += 1
                # refine non-DKI assets and insights
                Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "0"]
                Insight_list = [insight for insight, isDKI in zip(Insight_list, IsDKI_list) if isDKI == "0"]

            if any(Insight_list) and random.random() < 0.5:
                Insight, indices = get_repeated_string_indices(Insight_list)
                if Insight:
                    #print("Doing generation with insight for the non-DKI assets.")
                    Insight_Asset_list = [Asset_list[i] for i in indices]
                    Insight_Asset_list = list(set(Insight_Asset_list))
                    AssetCnt = len(Insight_Asset_list)
                    detail_insight_info = detail_info + "Insight: " + Insight + " \n"

                    message = construct_message(user_prompt_template, detail_insight_info, Insight_Asset_list, data_idx, AssetCnt, AssetType, FullLanguage)
                    #print("Insight message: ", message)
                    full_data_list.append(message)
                    data_idx += 1
                    data_withInsight += 1

            if any(Asset_list):
                # Doing generation without insight for the non-DKI assets
                Asset_list = list(set(Asset_list))
                AssetCnt = random.randint(1, len(Asset_list))
                rand_Asset_list = random.sample(Asset_list, AssetCnt)
                message = construct_message(user_prompt_template, detail_info, rand_Asset_list, data_idx, AssetCnt, AssetType, FullLanguage)
                #print("No Insight message: ", message)
                full_data_list.append(message)
                data_idx += 1

            if input_row % 10000 == 0:
                print("Processing row: ", input_row)
                print("Total data rows: ", data_idx)
                print("Total data with DKI: ", data_withDKI)
                print("Total data with Insight: ", data_withInsight)

            input_row += 1


    print("\nTotal input rows: ", input_row)
    print("Total data rows for model: ", data_idx)
    print("Total data with DKI: ", data_withDKI)
    print("Total data with Insight: ", data_withInsight)

    random.shuffle(full_data_list)
    train_size = int(len(full_data_list) * 0.8)
    train_data = full_data_list[:train_size]
    test_data = full_data_list[train_size:]
    print("Train data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    # output full data, train data, test data
    fw_full = open(args.FullData, 'w', encoding='utf-8')
    fw_train = open(args.train, 'w', encoding='utf-8')
    fw_test = open(args.test, 'w', encoding='utf-8')

    for row in full_data_list:
        row = json.dumps(row, ensure_ascii=False)
        fw_full.write(row + "\n")

    for row in train_data:
        row = json.dumps(row, ensure_ascii=False)
        fw_train.write(row + "\n")

    for row in test_data:
        row = json.dumps(row, ensure_ascii=False)
        fw_test.write(row + "\n")

    fw_full.close()
    fw_train.close()
    fw_test.close()

    # Output to parquet
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)
    df_train.to_parquet(args.train.replace(".tsv", ".parquet"))
    df_test.to_parquet(args.test.replace(".tsv", ".parquet"))


    print("Data generation is done.")

def ConvertTestToInferenceData(input_file, output_dir):
    df = pd.read_parquet(input_file)
    print(df.head())
    print(df.columns)
    print(df.shape)

    out_prompt_file = os.path.join(output_dir, "inference_prompt.tsv")
    out_response_file = os.path.join(output_dir, "inference_response.tsv")

    fw_prompt = open(out_prompt_file, 'w', encoding='utf-8')
    fw_response = open(out_response_file, 'w', encoding='utf-8')

    for index, row in df.iterrows():
        prompt_id = row['prompt_id']
        message = row['messages']
        for me_json in message:
            if me_json['role'] == "user":
                user_prompt = me_json['content']
                prompt_json = {"prompt_id": prompt_id, "prompt": user_prompt}
                fw_prompt.write(json.dumps(prompt_json, ensure_ascii=False) + "\n")
            elif me_json['role'] == "assistant":
                assistant_response = me_json['content']
                response_json = {"prompt_id": prompt_id, "response": assistant_response}
                fw_response.write(json.dumps(response_json, ensure_ascii=False) + "\n")

    fw_prompt.close()
    fw_response.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GenerateTrainTestDataForLLM')
    parser.add_argument('-i', '--input', help='input file', default="../data/AssetGeneration/OriginalCombinedAssets.tsv")
    #parser.add_argument('-i', '--input', help='input file', default="../data/AssetGeneration/test.tsv")
    parser.add_argument('-fu', '--FullData', help='tsv file', default="../data/AssetGeneration/FullData.tsv")
    parser.add_argument('-tr', '--train', help='tsv file', default="../data/AssetGeneration/train.tsv")
    parser.add_argument('-te', '--test', help='tsv file', default="../data/AssetGeneration/test.tsv")
    args = parser.parse_args()
    #main(args)
    ConvertTestToInferenceData(args.test.replace(".tsv", ".parquet"), "../data/AssetGeneration/")


#!/usr/bin/env python
#-*-coding:utf-8-*-


import tensorflow as tf
import pickle
import numpy as np
from tensorflow.contrib.crf import viterbi_decode

def decode(logits, trans, sequence_lengths, tag_num):
    viterbi_sequences = []
    logits = logits[0]
    print(logits)
    for logit, length in zip(logits, sequence_lengths):
        print(logit)
        viterbi_seq, viterbi_score = viterbi_decode(logit, trans)
        viterbi_sequences += viterbi_seq[:length]

    return viterbi_sequences


class Predictor(object):
    def __init__(self, map_file, checkpoint_dir):
        with open(map_file, "rb") as f:
            char_to_id,  tag_to_id, id_to_tag = pickle.load(f)

        id_to_char = {v:k for k,v in char_to_id.items()}
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.tag_num = len(tag_to_id)

        self.tf_config = tf_config
        self.char_to_id = char_to_id
        self.id_to_tag = {v:k for k, v in tag_to_id.items()}
        self.id_to_char = id_to_char
        self.checkpoint_dir = checkpoint_dir

        self.graph = tf.Graph()
        self.checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(self.checkpoint_file)
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement = True,
                log_device_placement = False )
            self.sess = tf.Session(config=session_conf)

            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)
                for op in self.graph.get_operations():

                    if "unflat_scores" in op.name:
                        print(op.name)

                self.char_inputs = self.graph.get_operation_by_name("input_x").outputs[0]
                self.lengths = self.graph.get_operation_by_name("sequence_lengths").outputs[0]
                # self.dropout = self.graph.get_operation_by_name("dropout").outputs[0]
                self.trans = self.graph.get_operation_by_name("transitions").outputs[0].eval()
                self.logits = self.graph.get_operation_by_name("predictions/predict").outputs[0]
                # self.pred = self.graph.get_operation_by_name("project/output/pred").outputs[0]


    def predict(self, text):
        char_id_list = []

        start = time.time()

        for w in list(text):
            if w in self.char_to_id:
                char_id_list.append(self.char_to_id.get(w))
            else:
                char_id_list.append(self.char_to_id["<OOV>"])

        input_x = np.array(char_id_list, dtype=np.int32).reshape(1, len(char_id_list))
        text_len = len(text)
        length_array = np.array([text_len]).reshape(1, -1)

        feed_dict = {
            self.char_inputs: input_x,
            self.lengths: length_array,
            # self.dropout: 1.0
        }

        logits = self.sess.run([self.logits], feed_dict=feed_dict)

        path = decode(logits, self.trans, [input_x.shape[1]], self.tag_num)
        tags = [self.id_to_tag[p] for p in path]
        print("cost : ", time.time() - start)
        return tags


if __name__ == '__main__':
    checkpoint_dir = "./results/seg_cnn/ckpt"
    map_file = "./datasets/seg/maps.pkl"
    predictor = Predictor(map_file=map_file, checkpoint_dir=checkpoint_dir)

    text = "北京欢迎你"
    rtext = "中共中央总书记、国家主席、中央军委主席、中央全面深化改革领导小组组长习近平1月23日下午主持召开中央全面深化改革领导小组第二次会议并发表重要讲话。他强调，2018年是贯彻党的十九大精神的开局之年，也是改革开放40周年，做好改革工作意义重大。要弘扬改革创新精神，推动思想再解放改革再深入工作再抓实，凝聚起全面深化改革的强大力量，在新起点上实现新突破。李克强、张高丽、汪洋、王沪宁出席会议。会议审议通过了《中央有关部门贯彻实施党的十九大〈报告〉重要改革举措分工方案》、《中央全面深化改革领导小组2018年工作要点》、《中央全面深化改革领导小组2017年工作总结报告》。会议审议通过了《关于推进社会公益事业建设领域政府信息公开的意见》、《关于提高技术工人待遇的意见》、《关于建立城乡居民基本养老保险待遇确定和基础养老金正常调整机制的指导意见》、《积极牵头组织国际大科学计划和大科学工程方案》、《关于推进孔子学院改革发展的指导意见》、《关于建立“一带一路”争端解决机制和机构的意见》、《关于改革完善仿制药供应保障及使用政策的若干意见》、《科学数据管理办法》、《知识产权对外转让有关工作办法（试行）》、《地方党政领导干部安全生产责任制规定》。会议还审议了《浙江省“最多跑一次”改革调研报告》。会议指出，推进社会公益事业建设领域政府信息公开，要准确把握社会公益事业建设规律和特点，加大信息公开力度，明确公开重点，细化公开内容，增强公开实效，提升社会公益事业透明度，推动社会公益资源配置更加公平公正，确保社会公益事业公益属性，维护社会公益事业公信力。会议强调，提高技术工人待遇，要坚持全心全意依靠工人阶级的方针，发挥政府、企业、社会协同作用，完善技术工人培养、评价、使用、激励、保障等措施，实现技高者多得、多劳者多得，增强技术工人职业荣誉感、自豪感、获得感，激发技术工人积极性、主动性、创造性。会议指出，建立城乡居民基本养老保险待遇确定和基础养老金正常调整机制，要按照兜底线、织密网、建机制的要求，建立激励约束有效、筹资权责清晰、保障水平适度的待遇确定和基础养老金正常调整机制，推动城乡居民基本养老保险待遇水平随经济发展逐步提高，确保参保居民共享经济社会发展成果。会议强调，牵头组织国际大科学计划和大科学工程，要按照国家创新驱动发展战略要求，以全球视野谋划科技开放合作，聚焦国际科技界普遍关注、对人类社会发展和科技进步影响深远的研究领域，集聚国内外优秀科技力量，量力而行、分步推进，形成一批具有国际影响力的标志性科研成果，提升我国战略前沿领域创新能力和国际影响力。会议指出，推进孔子学院改革发展，要围绕建设中国特色社会主义文化强国，服务中国特色大国外交，深化改革创新，完善体制机制，优化分布结构，加强力量建设，提高办学质量，使之成为中外人文交流的重要力量。会议强调，建立“一带一路”争端解决机制和机构，要坚持共商共建共享原则，依托我国现有司法、仲裁和调解机构，吸收、整合国内外法律服务资源，建立诉讼、调解、仲裁有效衔接的多元化纠纷解决机制，依法妥善化解“一带一路”商贸和投资争端，平等保护中外当事人合法权益，营造稳定、公平、透明的法治化营商环境。会议指出，改革完善仿制药供应保障及使用政策，要从群众需求出发，把临床必需、疗效确切、供应短缺、防治重大传染病和罕见病、处置突发公共卫生事件、儿童用药等作为重点，促进仿制药研发创新，提升质量疗效，提高药品供应保障能力，更好保障广大人民群众用药需求。会议强调，加强和规范科学数据管理，要适应大数据发展形势，积极推进科学数据资源开发利用和开放共享，加强重要数据基础设施安全保护，依法确定数据安全等级和开放条件，建立数据共享和对外交流的安全审查机制，为政府决策、公共安全、国防建设、科学研究提供有力支撑。会议指出，知识产权对外转让，要坚持总体国家安全观，依据现有法律法规和工作机制，对单位或者个人将其境内知识产权转让给外国企业、个人或者其他组织，严格审查范围、审查内容、审查机制，加强对涉及国家安全的知识产权对外转让行为的严格管理。会议强调，实行地方党政领导干部安全生产责任制，要坚持党政同责、一岗双责、齐抓共管、失职追责，牢固树立发展决不能以牺牲安全为代价的红线意识，明确地方党政领导干部主要安全生产职责，综合运用巡查督查、考核考察、激励惩戒等措施，强化地方各级党政领导干部“促一方发展、保一方平安”的政治责任。会议指出，党的十八大以来，浙江等地针对群众反映突出的办事难、投诉举报难等问题，从优化审批流程入手，推动实施“最多跑一次”改革，取得积极成效。各地区要结合实际，善于从基层和群众关心的问题上找出路、找办法，加大体制机制创新，以实际行动增强群众对改革的获得感。会议强调，2018年是站在新的历史起点上接力探索、接续奋进的关键之年，要全面贯彻党的十九大精神，以习近平新时代中国特色社会主义思想为指导，统筹推进党的十八大以来部署的改革举措和党的十九大部署的改革任务，更加注重改革的系统性、整体性、协同性，着力补齐重大制度短板，着力抓好改革任务落实，着力巩固拓展改革成果，着力提升人民群众获得感，不断将改革推深做实，推进基础性关键领域改革取得实质性成果。改革要突出重点，攻克难点，在破除各方面体制机制弊端、调整深层次利益格局上再拿下一些硬任务，重点推进国企国资、垄断行业、产权保护、财税金融、乡村振兴、社会保障、对外开放、生态文明等关键领域改革。要提高政治站位，勇于推进改革，敢于自我革命。要结合实际，实事求是，多从基层和群众关心的问题上找突破口，多推有地方特点的改革。要鼓励基层创新，继续发扬敢闯敢试、敢为人先的精神，推动形成更加浓厚、更有活力的改革创新氛围。要拿出实实在在的举措克服形式主义问题。主要负责同志要带好头，把责任和工作抓实，越是难度大、见效慢的越要抓在手上，不弃微末，不舍寸功。要把调查研究突出出来，把存在的矛盾和困难摸清摸透，把工作做实做深做好。改革督察要扩点拓面、究根探底，既要听其言，也要观其行、查其果。对不作为的，要抓住典型，严肃问责。中央全面深化改革领导小组成员出席，有关中央领导同志以及中央和国家机关有关部门负责同志列席会议。";

    # text= ""
    import time
    labels = predictor.predict(text)
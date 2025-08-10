def generate_mammogram_description(
    laterality,
    view_position,
    breast_density,
    breast_birads,
    finding_categories,
    finding_birads,
    width,
    height,
    xmin,
    ymin,
    xmax,
    ymax
):
    xmin = max(int(xmin / width * 1000) - 20, 0)
    ymin = max(int(ymin / height * 1000) - 20, 0)
    xmax = min(int(xmax / width * 1000) + 20, 1000)
    ymax = min(int(ymax / height * 1000) + 20, 1000)

    density_descriptions = {
        "DENSITY A": "图像整体灰度较低，乳房主要由低密度的脂肪组织构成，呈均匀的暗灰色，组织结构清晰，易于识别病灶。",
        "DENSITY B": "图像中可见散布的高亮区域，代表部分致密的腺体组织；整体灰度对比适中，部分区域组织结构轻微重叠。",
        "DENSITY C": "图像中大部分区域显示为亮灰色或高亮区域，乳腺组织密度较高，可能影响小病灶的可视化，需增强对局部结构的关注。",
        "DENSITY D": "图像普遍呈现高对比和高亮度，乳腺组织极其致密，组织结构层层叠加，严重遮挡潜在病灶，病灶识别难度大。"
    }

    birads_descriptions = {
        "BI-RADS 0": "图像信息不足或存在技术因素限制，需要进行额外成像检查（如放大视图、超声、MRI）以明确观察区域。",
        "BI-RADS 1": "图像未见可疑异常结构，腺体组织分布均匀，无肿块、钙化或结构扭曲，属于正常影像表现。",
        "BI-RADS 2": "图像中出现明确良性的结构，如单纯囊肿、脂肪瘤、粗大圆形钙化等，边界清晰、形态规则、无恶性特征。",
        "BI-RADS 3": "图像中可见轻度异常结构，如形态较规则但需随访的密度影或分散分布的钙化灶，恶性概率极低（<2%）。",
        "BI-RADS 4": "图像存在边界模糊、形状不规则的肿块或可疑钙化灶，需组织学检查进一步明确，恶性概率为2%至95%。",
        "BI-RADS 5": "图像中可见具有典型恶性特征的病灶，如分叶状肿块、高密度钙化簇或结构扭曲，恶性可能性极高（>95%）。",
        "BI-RADS 6": "图像中病灶已通过病理确诊为恶性，常用于治疗后影像随访或术前评估，需密切对比结构变化。"
    }

    category_descriptions = {
        "Mass": (
            "在CLAHE增强后的图像中，肿块（Mass）通常表现为与周围组织对比明显的亮灰色或白色区域，"
            "形态可能为圆形、椭圆形或不规则，边缘可以清晰、模糊或呈分叶状。"
            "大小从数毫米至数厘米不等，可能单发或多发，常位于局部密度较高区域。"
            "需重点观察边缘清晰度、内部回声均匀性及周围组织结构变化，以判断良恶性。"
        ),
        "Suspicious Calcification": (
            "钙化灶（Suspicious Calcification）在CLAHE图像中表现为密集的亮白色小点，"
            "常呈成簇分布，形态细小、不规则，可呈线状、分支状或粉末状。"
            "这些钙化灶多位于致密腺体区域，易被背景遮挡，需借助增强对比后的图像清晰识别其排列形态和密集程度，"
            "形态越不规则、分布越集中，恶性概率越高。"
        )
    }

    view_text = '头尾向（CC）' if view_position == 'CC' else '内外斜向（MLO）'
    side_text = '左侧' if laterality == 'L' else '右侧'
    density_description = density_descriptions.get(breast_density, "密度未明确。")
    breast_birads_desc = birads_descriptions.get(breast_birads, "未定义。")
    finding_birads_desc = birads_descriptions.get(finding_birads, "未定义。")

    if isinstance(finding_categories, list):
        finding_cat = finding_categories[0]
    elif isinstance(finding_categories, str):
        finding_cat = finding_categories.strip("[]'\" ")
    else:
        finding_cat = str(finding_categories)

    category_description = category_descriptions.get(finding_cat, "未识别的病变类型。")
    description = f"""该乳腺X线图像为{side_text}乳房（Laterality: {laterality}），拍摄视角为 {view_position}（{view_text}）。乳腺密度：{breast_density} – {density_description} 总体 BI-RADS 等级：{breast_birads} – {breast_birads_desc} 检测到的病变类型：{finding_cat} – {category_description} 病变 BI-RADS 等级：{finding_birads} – {finding_birads_desc} 图像尺寸：{width} × {height} 像素。病变框选区域 <ref>病变位置</ref><box>({xmin},{ymin}),({xmax},{ymax})</box>。"""
    return description

def generate_request_description():
    return """请严格标注并框选图像中所有细小、圆形的可疑肿块（Mass）或可疑钙化灶（Suspicious Calcification）区域，输出对应的检测框坐标，以便后续诊断分析, 检测框应紧贴所检测到的目标。"""
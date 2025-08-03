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
    xmin = int(xmin / width * 1000)
    ymin = int(ymin / height * 1000)
    xmax = int(xmax / width * 1000)
    ymax = int(ymax / height * 1000)

    density_descriptions = {
        "DENSITY A": "乳房几乎完全由脂肪组织组成。",
        "DENSITY B": "乳房中散布有部分致密组织。",
        "DENSITY C": "乳腺组织密度相对较高。",
        "DENSITY D": "乳腺组织非常致密。"
    }

    birads_descriptions = {
        "BI-RADS 0": "需要进一步成像检查。",
        "BI-RADS 1": "未见明显异常。",
        "BI-RADS 2": "发现良性病变。",
        "BI-RADS 3": "存在轻微但不明确的异常。",
        "BI-RADS 4": "发现可疑病变，需要进一步评估。",
        "BI-RADS 5": "高度怀疑恶性病变。",
        "BI-RADS 6": "已确诊的恶性病变。"
    }

    category_descriptions = {
        "Mass": (
            "肿块（Mass）表现为图像中的圆形或椭圆形病灶，"
            "边缘可能清晰、模糊或呈分叶状，大小和形状多变。"
            "常见于局部密度异常区域，需结合BI-RADS等级进一步评估其良恶性。"
        ),
        "Suspicious Calcification": (
            "可疑钙化灶（Suspicious Calcification）通常表现为图像中的细小、高密度白点，"
            "多成簇分布，形态不规则，可能呈线状或树枝状排列。"
            "这些钙化灶往往难以辨识，需重点观察其分布特征和密集程度。"
        ),
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


print(generate_mammogram_description(
    laterality="L",
    view_position="CC",
    breast_density="DENSITY C",
    breast_birads="BI-RADS 3",
    finding_categories=["Mass"],
    finding_birads="BI-RADS 4",
    width=2800,
    height=3500,
    xmin=400,
    ymin=1500,
    xmax=800,
    ymax=2000
))

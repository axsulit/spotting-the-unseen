import Image from "next/image";

export default function Problem() {
	return (
		<section className="py-16 px-20 bg-white">
			<div className="container">
				<div className="space-y-4">
					<p className="leading-relaxed">
						Recent face forgery detection methods enhance performance by adding complex modules or multi-branch networks, but these improvements often come with high computational cost and limited scalability. Such demands restrict deployment on real-world systems where media quality varies and resources are constrained.
					</p>
					<p className="leading-relaxed">
						In this work, we conduct a comparative analysis of leading spatial-, frequency-, and attention-based detection models to evaluate their robustness under perturbations, compression, and cross-dataset conditions. Our study identifies the strengths and vulnerabilities of each approach and reveals the architectural factors that most influence generalization and efficiency. More importantly, our findings provide practical guidance for designing more reliable and deployable deepfake detection frameworks.
					</p>
				</div>
			</div>
			
			<div className="w-full mt-8">
				<Image
					src="/figure_1.png"
					alt="Problem"
					width={1920}
					height={1080}
					className="w-full h-auto rounded object-contain"
				/>
				<p className="mt-4 text-center text-sm  mx-auto">
					Figure: Average F1-scores of spatial (Xception, RECCE), frequency-based (FreqNet, HiFi-FD), and attention-based (Multi-Att, RFM) face forgery detection models evaluated under a wide range of perturbations. Each radar chart illustrates model robustness across resolution resizing, blurring, varying noise levels, color mismatch, and boundary splicing.
				</p>
			</div>
		</section>
	)
}
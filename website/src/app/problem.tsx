import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Problem() {
	return (
		<section className="py-16 px-4 bg-white">
				<div className="container mx-auto max-w-4xl">
					<Card className="border-0 shadow-lg">
						<CardHeader>
							<CardTitle className="text-2xl text-gray-900">
								The Problem
							</CardTitle>
							<CardDescription className="text-lg">
								Understanding the challenges in deepfake
								detection
							</CardDescription>
						</CardHeader>
						<CardContent className="text-gray-700 leading-relaxed">
							<p>
								The rise of deepfake technologies has created
								pressing concerns about misinformation and the
								integrity of digital media. Current face forgery
								detection models struggle under real-world
								conditions like resolution degradation,
								compression artifacts, and unseen forgery
								techniques. This research investigates how well
								state-of-the-art models perform across these
								challenging scenarios.
							</p>
						</CardContent>
					</Card>
				</div>
			</section>
	)
}
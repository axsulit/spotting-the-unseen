import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";
import { toast } from "sonner";

export default function Citation() {
	return (
		<section className="py-16 px-4 bg-gray-50">
				<div className="container mx-auto max-w-4xl">
					<Card className="border-0 shadow-lg">
						<CardHeader>
							<CardTitle className="text-2xl text-gray-900">
								Cite our Paper
							</CardTitle>
						</CardHeader>
						<CardContent>
							<div className="bg-gray-100 p-4 rounded-lg font-mono text-sm text-gray-800 leading-relaxed">
								<p>
									Exconde, I. R. C., Gon Gon, Z. A. F., Sulit,
									A. G. M., & Torio, Y. D. (2025). Spotting
									the unseen: A comprehensive analysis of face
									forgery detection models. Center for
									Computational Imaging & Visual Innovations,
									De La Salle University.
								</p>
							</div>
							<div className="flex flex-wrap gap-4 mt-6">
								<Button
									variant="outline"
									className="bg-white"
									onClick={() => {
										navigator.clipboard
											.writeText(
												`@inproceedings{exconde2025spotting,
                  author    = {Exconde, I. R. C., Gon Gon, Z. A. F., Sulit, A. G. M., & Torio, Y. D.},
                  title     = {Spotting the Unseen: A Comprehensive Analysis of Face Forgery Detection Models},
                  year      = {2025},
                  institution = {De La Salle University},
                  }
                  `,
											)
											.then(() => {
												toast.success(
													'Copied to clipboard',
												);
											})
											.catch(() => {
												toast.error('Failed to copy');
											});
									}}
								>
									<FileText className="w-4 h-4 mr-2" />
									BibTeX
								</Button>
								{/* <Button variant="outline" className="bg-white">
                  <ExternalLink className="w-4 h-4 mr-2" />
                  DOI
                </Button> */}
							</div>
						</CardContent>
					</Card>
				</div>
			</section>
	)
}
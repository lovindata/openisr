import { LinkAtm } from "@/v2/features/shared/components/atoms/LinkAtm";
import { SvgIconAtm } from "@/v2/features/shared/components/atoms/SvgIconAtm";
import { useNavigate } from "react-router-dom";

export default function NavBarOrg() {
  const navigate = useNavigate();

  return (
    <nav className="max-md:h-12 md:w-10">
      <div
        className="fixed flex items-center justify-between border-white border-opacity-50
        bg-black max-md:w-full max-md:flex-row max-md:border-b max-md:px-3 max-md:py-2 md:h-full md:flex-col md:border-r md:px-2 md:py-3"
      >
        <SvgIconAtm
          type="logo"
          className="h-6 w-6 cursor-pointer"
          onClick={() => navigate("/app")}
        />
        <div className="flex flex-col items-center max-md:flex-row max-md:space-x-3 md:space-y-3">
          <LinkAtm href="/app">
            <SvgIconAtm type="donation" className="h-5 w-5" />
          </LinkAtm>
          <LinkAtm href="https://hub.docker.com/u/ilovedatajjia">
            <SvgIconAtm type="docker" className="h-5 w-5" />
          </LinkAtm>
          <LinkAtm href="https://github.com/iLoveDataJjia">
            <SvgIconAtm type="github" className="h-5 w-5" />
          </LinkAtm>
        </div>
      </div>
    </nav>
  );
}

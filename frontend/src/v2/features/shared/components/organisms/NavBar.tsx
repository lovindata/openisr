import { Link } from "@/v2/features/shared/components/atoms/Link";
import { SvgIcon } from "@/v2/features/shared/components/atoms/SvgIcon";
import { useNavigate } from "react-router-dom";

export default function NavBar() {
  const navigate = useNavigate();

  return (
    <nav className="max-md:h-12 md:w-10">
      <div
        className="fixed flex items-center justify-between border-white border-opacity-50
        bg-black max-md:w-full max-md:flex-row max-md:border-b max-md:px-3 max-md:py-2 md:h-full md:flex-col md:border-r md:px-2 md:py-3"
      >
        <SvgIcon
          type="logo"
          className="h-6 w-6 cursor-pointer"
          onClick={() => navigate("/app")}
        />
        <div className="flex flex-col items-center max-md:flex-row max-md:space-x-3 md:space-y-3">
          <Link href="/app">
            <SvgIcon type="donation" className="h-5 w-5" />
          </Link>
          <Link href="https://hub.docker.com/u/ilovedatajjia">
            <SvgIcon type="docker" className="h-5 w-5" />
          </Link>
          <Link href="https://github.com/iLoveDataJjia">
            <SvgIcon type="github" className="h-5 w-5" />
          </Link>
        </div>
      </div>
    </nav>
  );
}
